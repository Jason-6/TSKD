#!/usr/bin/env bash

# Copyright 2020 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                AISHELL-1                                 "
echo ============================================================================

stage=4
stop_stage=4
gpu=1
benchmark=true
deterministic=false
pin_memory=false
stdout=false
wandb_id=""
corpus=aishell1

### vocabulary
unit=char

#########################
# ASR configuration
#########################
conf=conf/asr/paper/lambda/lstm/blstm_las_6.yaml
conf2=
asr_init=
external_lm=

#########################
# LM configuration
#########################
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
feature_data=/vol/nfs/mgl/datasets/aishell_data
model=./data/results/${corpus}

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
data=/vol/nfs/mgl/datasets


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


set -e
set -u
set -o pipefail

train_set=train_sp
dev_set=dev_sp
test_set="test_sp"

use_wandb=false
if [ ! -z ${wandb_id} ]; then
    use_wandb=true
    wandb login ${wandb_id}
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    mkdir -p ${data}
    local/download_and_untar.sh ${data} "www.openslr.org/resources/33" data_aishell
    local/download_and_untar.sh ${data} "www.openslr.org/resources/33" resource_aishell
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${feature_data}/.done_stage_0 ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    local/aishell_data_prep.sh ${data}/data_aishell_pre_train_tune/wav ${data}/data_aishell_pre_train_tune/transcript ${feature_data}
    # remove space in text
    for x in train dev test; do
        cp ${feature_data}/${x}/text ${feature_data}/${x}/text.org
        paste -d " " <(cut -f 1 -d" " ${feature_data}/${x}/text.org) <(cut -f 2- -d" " ${feature_data}/${x}/text.org | tr -d " ") \
            > ${feature_data}/${x}/text
        rm ${feature_data}/${x}/text.org
    done

    touch ${feature_data}/.done_stage_0 && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${feature_data}/.done_stage_1 ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    for x in train dev test; do
        steps/make_fbank.sh --nj 32 --cmd "$train_cmd" --write_utt2num_frames true \
            ${feature_data}/${x} ${feature_data}/log/make_fbank/${x} ${feature_data}/fbank || exit 1;
        utils/fix_data_dir.sh ${feature_data}/${x}
    done

    speed_perturb_3way.sh ${feature_data} train ${train_set}
    cp -rf ${feature_data}/dev ${feature_data}/${dev_set}
    cp -rf ${feature_data}/test ${feature_data}/${test_set}

    # Compute global CMVN
    compute-cmvn-stats scp:${feature_data}/${train_set}/feats.scp ${feature_data}/${train_set}/cmvn.ark || exit 1;

    # Apply global CMVN & dump features
    dump_feat.sh --cmd "$train_cmd" --nj 80 \
        ${feature_data}/${train_set}/feats.scp ${feature_data}/${train_set}/cmvn.ark ${feature_data}/log/dump_feat/${train_set} ${feature_data}/dump/${train_set} || exit 1;
    for x in ${dev_set} ${test_set}; do
        dump_dir=${feature_data}/dump/${x}
        dump_feat.sh --cmd "$train_cmd" --nj 32 \
            ${feature_data}/${x}/feats.scp ${feature_data}/${train_set}/cmvn.ark ${feature_data}/log/dump_feat/${x} ${dump_dir} || exit 1;
    done

    touch ${feature_data}/.done_stage_1
fi

dict=${feature_data}/dict/${train_set}.txt; mkdir -p ${feature_data}/dict
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${feature_data}/.done_stage_2 ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    make_vocab.sh --unit ${unit} --speed_perturb true \
        ${feature_data} ${dict} ${feature_data}/${train_set}/text || exit 1;

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${feature_data}/dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        dump_dir=${feature_data}/dump/${x}
        make_dataset.sh --feat ${dump_dir}/feats.scp --unit ${unit} \
            ${feature_data}/${x} ${dict} > ${feature_data}/dataset/${x}.tsv || exit 1;
    done

    touch ${feature_data}/.done_stage_2 && echo "Finish creating dataset for ASR (stage: 2)."
fi

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)


mkdir -p ${model}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    export OMP_NUM_THREADS=${n_gpus}
    python \
        ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --train_set ${feature_data}/dataset/${train_set}.tsv \
        --dev_set ${feature_data}/dataset/${dev_set}.tsv \
        --eval_sets ${feature_data}/dataset/${test_set}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume}

    echo "Finish LM training (stage: 3)."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    export OMP_NUM_THREADS=${n_gpus}
#    conda list/home/mgl/github_code/neural_sp/neural_sp/models/seq2seq/__init___.py
    python \
        ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py --local_world_size=${n_gpus} \
        --corpus ${corpus} \
        --use_wandb ${use_wandb} \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --cudnn_deterministic ${deterministic} \
        --pin_memory ${pin_memory} \
        --train_set ${feature_data}/dataset/${train_set}.tsv \
        --dev_set ${feature_data}/dataset/${dev_set}.tsv \
        --unit ${unit} \
        --dict ${dict} \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --stdout ${stdout} \
        --resume ${resume}\
        --remove_old_checkpoints False || exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
