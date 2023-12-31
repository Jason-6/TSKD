# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Args options for the ASR task."""

import configargparse
from distutils.util import strtobool
import logging
from omegaconf import OmegaConf
import os

from neural_sp.bin.train_utils import load_config
from neural_sp.bin.args_common import add_args_common

ENCODER_TYPES = ['blstm', 'lstm', 'bgru', 'gru',
                 'conv_blstm', 'conv_lstm', 'conv_bgru', 'conv_gru',
                 'transformer', 'conv_transformer', 'conv_uni_transformer',
                 'conformer', 'conv_conformer', 'conv_uni_conformer',
                 'conformer_v2', 'conv_conformer_v2', 'conv_uni_conformer_v2',
                 'tds', 'gated_conv']

DECODER_TYPES = ['lstm', 'gru', 'transformer',
                 'lstm_transducer', 'gru_transducer',
                 'asg']

logger = logging.getLogger(__name__)


def parse_args_train(input_args):
    parser = build_parser()
    user_args = parser.parse_known_args(input_args)[0]

    config = OmegaConf.load(user_args.config)
    if user_args.config2 is not None:
        config = OmegaConf.merge(config, OmegaConf.load(user_args.config2))

    # register module specific arguments
    # encoder
    parser = register_args_encoder(parser, user_args, user_args.enc_type)
    user_args = parser.parse_known_args(input_args)[0]  # to avoid args conflict
    # decoder
    parser = register_args_decoder(parser, user_args, user_args.dec_type)
    # auxiliary decoders
    if user_args.dec_config_sub1 is not None and user_args.dec_type != config.dec_config_sub1.dec_type:
        user_args = parser.parse_known_args(input_args)[0]  # to avoid args conflict
        parser = register_args_decoder(parser, user_args, config.dec_config_sub1.dec_type)
    user_args = parser.parse_args()

    # merge to omegaconf
    for k, v in vars(user_args).items():
        if k not in config:
            config[k] = v

    return config


def parse_args_eval(input_args):
    parser = build_parser()
    user_args, _ = parser.parse_known_args(input_args)

    # Load a yaml config file
    dir_name = os.path.dirname(user_args.recog_model[0])
    config = load_config(os.path.join(dir_name, 'conf.yml'))

    # register module specific arguments to support new args after training
    # encoder
    parser = register_args_encoder(parser, user_args, config.enc_type)
    user_args, _ = parser.parse_known_args(input_args)  # to avoid args conflict
    # decoder
    if config.dec_config_sub1 is not None:
        user_args.dec_config_sub1 = config.dec_config_sub1
    parser = register_args_decoder(parser, user_args, config.dec_type)
    if config.dec_config_sub1 is not None and config.dec_type != config.dec_config_sub1.dec_type:
        user_args, _ = parser.parse_known_args(input_args)  # to avoid args conflict
        parser = register_args_decoder(parser, user_args, config.dec_config_sub1.dec_type)
    user_args = parser.parse_args()

    # Overwrite to omegaconf
    for k, v in vars(user_args).items():
        if 'recog' in k or k not in config:
            config[k] = v
            logger.info('Overwrite configration: %s => %s' % (k, v))

    return config, dir_name


def register_args_encoder(parser, args, enc_type):
    if enc_type == 'tds':
        from neural_sp.models.seq2seq.encoders.tds import TDSEncoder as module
    elif enc_type == 'gated_conv':
        from neural_sp.models.seq2seq.encoders.gated_conv import GatedConvEncoder as module
    elif 'transformer' in enc_type:
        from neural_sp.models.seq2seq.encoders.transformer import TransformerEncoder as module
    elif 'conformer' in enc_type:
        from neural_sp.models.seq2seq.encoders.conformer import ConformerEncoder as module
    else:
        from neural_sp.models.seq2seq.encoders.rnn import RNNEncoder as module
    if hasattr(module, 'add_args'):
        parser = module.add_args(parser, args)
    return parser


def register_args_decoder(parser, args, dec_type):
    if dec_type in ['transformer']:
        from neural_sp.models.seq2seq.decoders.transformer import TransformerDecoder as module
    elif dec_type in ['lstm_transducer', 'gru_transducer']:
        from neural_sp.models.seq2seq.decoders.rnn_transducer import RNNTransducer as module
    elif dec_type == 'asg':
        from neural_sp.models.seq2seq.decoders.asg import ASGDecoder as module
    else:
        from neural_sp.models.seq2seq.decoders.las import RNNDecoder as module
    if hasattr(module, 'add_args'):
        parser = module.add_args(parser, args)
    return parser


def build_parser():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser = add_args_common(parser)
    # general
    parser.add('--config2', is_config_file=True, default=None, nargs='?',
               help='another config file path to overwrite --config')
    # dataset
    parser.add_argument('--train_set_sub1', type=str, default=False,
                        help='tsv file path for the training set for the 1st auxiliary task')
    parser.add_argument('--train_set_sub2', type=str, default=False,
                        help='tsv file path for the training set for the 2nd auxiliary task')
    parser.add_argument('--train_word_alignment', type=str,
                        help='word alignment directory path for the training set')
    parser.add_argument('--train_ctc_alignment', type=str,
                        help='CTC alignment directory path for the training set')
    parser.add_argument('--dev_set_sub1', type=str, default=False,
                        help='tsv file path for the development set for the 1st auxiliary task')
    parser.add_argument('--dev_set_sub2', type=str, default=False,
                        help='tsv file path for the development set for the 2nd auxiliary task')
    parser.add_argument('--dev_word_alignment', type=str,
                        help='word alignment directory path for the development set')
    parser.add_argument('--dev_ctc_alignment', type=str,
                        help='CTC alignment directory path for the development set')
    parser.add_argument('--dict_sub1', type=str, default=False,
                        help='dictionary file path for the 1st auxiliary task')
    parser.add_argument('--dict_sub2', type=str, default=False,
                        help='dictionary file path for the 2nd auxiliary task')
    parser.add_argument('--unit_sub1', type=str, default=False,
                        choices=['wp', 'char', 'phone'],
                        help='output unit for the 1st auxiliary task')
    parser.add_argument('--unit_sub2', type=str, default=False,
                        choices=['wp', 'char', 'phone'],
                        help='output unit for the 2nd auxiliary task')
    parser.add_argument('--wp_model_sub1', type=str, default=False, nargs='?',
                        help='wordpiece model path for the 1st auxiliary task')
    parser.add_argument('--wp_model_sub2', type=str, default=False, nargs='?',
                        help='wordpiece model path for the 2nd auxiliary task')
    # features
    parser.add_argument('--input_type', type=str, default='speech',
                        choices=['speech', 'text'],
                        help='type of input features')
    parser.add_argument('--n_splices', type=int, default=1,
                        help='number of input frames to splice (both for left and right frames)')
    parser.add_argument('--n_stacks', type=int, default=1,
                        help='number of input frames to stack (frame stacking)')
    parser.add_argument('--n_skips', type=int, default=1,
                        help='number of input frames to skip')
    parser.add_argument('--max_n_frames', type=int, default=2000,
                        help='maximum number of input frames')
    parser.add_argument('--min_n_frames', type=int, default=40,
                        help='minimum number of input frames')
    parser.add_argument('--dynamic_batching', type=strtobool, default=True,
                        help='')
    parser.add_argument('--input_noise_std', type=float, default=0,
                        help='standard deviation of Gaussian noise to input features')
    parser.add_argument('--weight_noise_std', type=float, default=0,
                        help='standard deviation of Gaussian noise to weight parameters')
    parser.add_argument('--sequence_summary_network', type=strtobool, default=False,
                        help='use sequence summary network')
    # topology (encoder)
    parser.add_argument('--enc_type', type=str, default='blstm',
                        choices=ENCODER_TYPES,
                        help='type of the encoder')
    parser.add_argument('--enc_n_layers', type=int, default=5,
                        help='number of encoder RNN layers')
    parser.add_argument('--enc_n_layers_sub1', type=int, default=0,
                        help='number of encoder RNN layers in the 1st auxiliary task')
    parser.add_argument('--enc_n_layers_sub2', type=int, default=0,
                        help='number of encoder RNN layers in the 2nd auxiliary task')
    parser.add_argument('--subsample', type=str, default="1_1_1_1_1",
                        help='delimited list input')
    parser.add_argument('--subsample_type', type=str, default='drop',
                        choices=['drop', 'concat', 'max_pool', 'mean_pool', 'conv1d', 'add'],
                        help='type of subsampling in the encoder')
    # topology (decoder)
    parser.add_argument('--dec_type', type=str, default='lstm',
                        choices=DECODER_TYPES,
                        help='type of the decoder')
    parser.add_argument('--dec_type_sub1', type=str, default='lstm',
                        choices=DECODER_TYPES,
                        help='type of the decoder in the 1st auxiliary task')
    parser.add_argument('--dec_type_sub2', type=str, default='lstm',
                        choices=DECODER_TYPES,
                        help='type of the decoder in the 2nd auxiliary task')
    parser.add_argument('--dec_config_sub1', default=None,
                        help='decoder configuration in the 1st auxiliary task')
    parser.add_argument('--dec_config_sub2', default=None,
                        help='decoder configuration in the 2nd auxiliary task')
    parser.add_argument('--dec_n_layers', type=int, default=1,
                        help='number of decoder RNN layers')
    parser.add_argument('--tie_embedding', type=strtobool, default=False, nargs='?',
                        help='tie weights of an embedding matrix and a linear layer before the softmax layer')
    parser.add_argument('--ctc_fc_list', type=str, default="", nargs='?',
                        help='')
    # optimization
    parser.add_argument('--batch_size_type', type=str, default='seq',
                        choices=['seq', 'token', 'frame'],
                        help='type of batch size counting')
    parser.add_argument('--metric', type=str, default='edit_distance',
                        choices=['edit_distance', 'loss', 'accuracy', 'ppl', 'bleu', 'mse'],
                        help='metric for evaluation during training')
    parser.add_argument('--sort_stop_epoch', type=int, default=10000,
                        help='epoch to stop soring utterances by length')
    parser.add_argument('--sort_short2long', type=strtobool, default=True,
                        help='sort utterances in the ascending order')
    parser.add_argument('--sort_by', type=str, default='input',
                        choices=['input', 'output', 'shuffle', 'utt_id'],
                        help='metric to sort utterances')
    parser.add_argument('--shuffle_bucket', type=strtobool, default=False,
                        help='gather the similar length of utterances and shuffle them')
    # initialization
    parser.add_argument('--asr_init', type=str, default=False, nargs='?',
                        help='pre-trained seq2seq model path')
    parser.add_argument('--asr_init_enc_only', type=strtobool, default=False,
                        help='Initialize the encoder only')
    parser.add_argument('--freeze_encoder', type=strtobool, default=False,
                        help='freeze the encoder parameter')
    # regularization
    parser.add_argument('--dropout_in', type=float, default=0.0,
                        help='dropout probability for the input')
    parser.add_argument('--dropout_enc', type=float, default=0.0,
                        help='dropout probability for the encoder')
    parser.add_argument('--dropout_dec', type=float, default=0.0,
                        help='dropout probability for the decoder')
    parser.add_argument('--dropout_emb', type=float, default=0.0,
                        help='dropout probability for the embedding')
    parser.add_argument('--dropout_att', type=float, default=0.0,
                        help='dropout probability for the attention weights')
    parser.add_argument('--ctc_lsm_prob', type=float, default=0.0,
                        help='probability of label smoothing for CTC')
    # SpecAugment
    parser.add_argument('--freq_width', type=int, default=27,
                        help='width of frequency mask for SpecAugment')
    parser.add_argument('--n_freq_masks', type=int, default=0,
                        help='number of frequency masks for SpecAugment')
    parser.add_argument('--time_width', type=int, default=100,
                        help='width of time mask for SpecAugment')
    parser.add_argument('--n_time_masks', type=int, default=0,
                        help='number of time masks for SpecAugment')
    parser.add_argument('--time_width_upper', type=float, default=1.0,
                        help='')
    parser.add_argument('--adaptive_number_ratio', type=float, default=0.0,
                        help='adaptive multiplicity ratio for time masking')
    parser.add_argument('--adaptive_size_ratio', type=float, default=0.0,
                        help='adaptive size ratio for time masking')
    parser.add_argument('--max_n_time_masks', type=int, default=20,
                        help='maximum number of time masking')
    # MTL
    parser.add_argument('--total_weight', type=float, default=1.0,
                        help='total loss weight')
    parser.add_argument('--ctc_weight', type=float, default=0.0,
                        help='CTC loss weight for the main task')
    parser.add_argument('--ctc_weight_sub1', type=float, default=0.0,
                        help='CTC loss weight for the 1st auxiliary task')
    parser.add_argument('--ctc_weight_sub2', type=float, default=0.0,
                        help='CTC loss weight for the 2nd auxiliary task')
    parser.add_argument('--sub1_weight', type=float, default=0.0,
                        help='total loss weight for the 1st auxiliary task')
    parser.add_argument('--sub2_weight', type=float, default=0.0,
                        help='total loss weight for the 2nd auxiliary task')
    parser.add_argument('--mtl_per_batch', type=strtobool, default=False, nargs='?',
                        help='change mini-batch per task')
    parser.add_argument('--task_specific_layer', type=strtobool, default=False, nargs='?',
                        help='insert a task-specific encoder layer per task')
    # forward-backward
    parser.add_argument('--bwd_weight', type=float, default=0.0,
                        help='cross entropy loss weight for the backward decoder in the main task')
    # cold fusion, LM initialization
    parser.add_argument('--external_lm', type=str, default=False, nargs='?',
                        help='LM path')
    parser.add_argument('--lm_fusion', type=str, default='',
                        choices=['', 'cold', 'cold_prob', 'deep', 'cold_attention'],
                        help='type of LM fusion')
    parser.add_argument('--lm_init', type=strtobool, default=False,
                        help='initialize the decoder with the external LM')
    # contextualization
    parser.add_argument('--discourse_aware', type=strtobool, default=False, nargs='?',
                        help='carry over the last decoder state to the initial state in the next utterance')
    # MBR
    parser.add_argument('--mbr_training', type=strtobool, default=False,
                        help='Minimum Bayes Risk (MBR) training')
    parser.add_argument('--mbr_ce_weight', type=float, default=0.0,
                        help='MBR loss weight for the main task')
    parser.add_argument('--mbr_nbest', type=int, default=4,
                        help='N-best for MBR training')
    parser.add_argument('--mbr_softmax_smoothing', type=float, default=0.8,
                        help='softmax smoothing (beta) for MBR training')
    # TransformerXL
    parser.add_argument('--bptt', type=int, default=0,
                        help='number of tokens to truncate in TransformerXL decoder during training')
    parser.add_argument('--mem_len', type=int, default=0,
                        help='number of tokens for memory in TransformerXL decoder during training')
    # distillation related
    parser.add_argument('--teacher', default=False, nargs='?',
                        help='Teacher ASR model for knowledge distillation')
    parser.add_argument('--teacher_lm', default=False, nargs='?',
                        help='Teacher LM for knowledge distillation')
    parser.add_argument('--distillation_weight', type=float, default=0.1,
                        help='soft label weight for knowledge distillation')
    # special label
    parser.add_argument('--replace_sos', type=strtobool, default=False,
                        help='')
    # decoding parameters
    parser.add_argument('--recog_word_alignments', type=str, default=[], nargs='+',
                        help='word alignment directory paths for the evaluation sets')
    parser.add_argument('--recog_first_n_utt', type=int, default=-1,
                        help='recognize the first N utterances for quick evaluation')
    parser.add_argument('--recog_model_bwd', type=str, default=False, nargs='?',
                        help='model path in the reverse direction')
    parser.add_argument('--recog_unit', type=str, default=False, nargs='?',
                        choices=['word', 'wp', 'char', 'phone', 'word_char', 'char_space'],
                        help='')
    parser.add_argument('--recog_metric', type=str, default='edit_distance',
                        choices=['edit_distance', 'loss', 'accuracy', 'ppl', 'bleu'],
                        help='metric for evaluation')
    parser.add_argument('--recog_oracle', type=strtobool, default=False,
                        help='recognize by teacher-forcing')
    parser.add_argument('--recog_beam_width', type=int, default=1,
                        help='size of beam')
    parser.add_argument('--recog_max_len_ratio', type=float, default=1.0,
                        help='')
    parser.add_argument('--recog_min_len_ratio', type=float, default=0.0,
                        help='')
    parser.add_argument('--recog_length_penalty', type=float, default=0.0,
                        help='length penalty')
    parser.add_argument('--recog_length_norm', type=strtobool, default=False, nargs='?',
                        help='normalize score by hypothesis length')
    parser.add_argument('--recog_coverage_penalty', type=float, default=0.0,
                        help='coverage penalty')
    parser.add_argument('--recog_coverage_threshold', type=float, default=0.0,
                        help='coverage threshold')
    parser.add_argument('--recog_gnmt_decoding', type=strtobool, default=False, nargs='?',
                        help='adopt Google NMT beam search decoding')
    parser.add_argument('--recog_eos_threshold', type=float, default=1.5,
                        help='threshold for emitting a EOS token')
    parser.add_argument('--recog_lm_weight', type=float, default=0.0,
                        help='weight of first-pass LM score')
    parser.add_argument('--recog_ilm_weight', type=float, default=0.0,
                        help='weight of internla LM score')
    parser.add_argument('--recog_lm_second_weight', type=float, default=0.0,
                        help='weight of second-pass LM score')
    parser.add_argument('--recog_lm_bwd_weight', type=float, default=0.0,
                        help='weight of second-pass backward LM score. \
                                  First-pass backward LM in case of synchronous bidirectional decoding.')
    parser.add_argument('--recog_cache_embedding', type=strtobool, default=True,
                        help='cache token emebdding')
    parser.add_argument('--recog_ctc_weight', type=float, default=0.0,
                        help='weight of CTC score')
    parser.add_argument('--recog_lm', type=str, default=False, nargs='?',
                        help='path to first-pass LM for shallow fusion')
    parser.add_argument('--recog_lm_second', type=str, default=False, nargs='?',
                        help='path to second-pass LM for rescoring')
    parser.add_argument('--recog_lm_bwd', type=str, default=False, nargs='?',
                        help='path to second-pass LM in the reverse direction for rescoring')
    parser.add_argument('--recog_resolving_unk', type=strtobool, default=False,
                        help='resolving UNK for the word-based model')
    parser.add_argument('--recog_fwd_bwd_attention', type=strtobool, default=False,
                        help='forward-backward attention decoding')
    parser.add_argument('--recog_bwd_attention', type=strtobool, default=False,
                        help='backward attention decoding')
    parser.add_argument('--recog_reverse_lm_rescoring', type=strtobool, default=False,
                        help='rescore with another LM in the reverse direction')
    parser.add_argument('--recog_asr_state_carry_over', type=strtobool, default=False,
                        help='carry over ASR decoder state')
    parser.add_argument('--recog_lm_state_carry_over', type=strtobool, default=False,
                        help='carry over LM state')
    parser.add_argument('--recog_softmax_smoothing', type=float, default=1.0,
                        help='softmax smoothing (beta) for diverse hypothesis generation')
    parser.add_argument('--recog_wordlm', type=strtobool, default=False,
                        help='')
    parser.add_argument('--recog_longform_max_n_frames', type=int, default=0,
                        help='maximum input length for long-form evaluation')
    parser.add_argument('--recog_streaming', type=strtobool, default=False,
                        help='streaming decoding (both encoding and decoding are streaming)')
    parser.add_argument('--recog_streaming_encoding', type=strtobool, default=False,
                        help='streaming encoding (decoding is offline)')
    parser.add_argument('--recog_block_sync', type=strtobool, default=False,
                        help='block-synchronous streaming beam search decoding')
    parser.add_argument('--recog_block_sync_size', type=int, default=40,
                        help='block size in block-synchronous streaming beam search decoding')
    parser.add_argument('--recog_ctc_spike_forced_decoding', type=strtobool, default=False,
                        help='force MoChA to generate tokens corresponding to CTC spikes')
    parser.add_argument('--recog_ctc_vad', type=strtobool, default=True,
                        help='')
    parser.add_argument('--recog_ctc_vad_blank_threshold', type=int, default=40,
                        help='')
    parser.add_argument('--recog_ctc_vad_spike_threshold', type=float, default=0.1,
                        help='')
    parser.add_argument('--recog_ctc_vad_n_accum_frames', type=int, default=4000,
                        help='')
    parser.add_argument('--recog_mma_delay_threshold', type=int, default=-1,
                        help='delay threshold for MMA decoder')
    parser.add_argument('--recog_mem_len', type=int, default=0,
                        help='number of tokens for memory in TransformerXL decoder during evaluation')
    parser.add_argument('--recog_rnnt_beam_search_type', type=str, default='time_sync_mono',
                        choices=['time_sync_mono', 'time_sync'],
                        help='beam search algorithm for RNN-T')
    parser.add_argument('--recog_mocha_p_choose_threshold', type=float, default=0.5,
                        help='threshold for p_choose during at test time')
    # pre train
    parser.add_argument('--start_hk_epoch', type=int, default=200,
                        help='start hk epoch')
    return parser
