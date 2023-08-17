from fairseq.modules.lightweight_convolution import LightweightConv1d
import torch.nn as nn

conv1d = nn.Conv1d(256, 256, 15, 1, 7, groups=256)
print(conv1d)
lightweightconv1d = LightweightConv1d(256, kernel_size=12, padding=7)

print(LightweightConv1d)
