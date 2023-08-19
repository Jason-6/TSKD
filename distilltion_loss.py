import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation(y, teacher_scores, temp, alpha):
    print(y.shape, teacher_scores.shape)
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha)


small_logits = np.load("/home/mgl/kd_ndarry/small_model.npy")
large_logits = np.load("/home/mgl/kd_ndarry/large_model.npy")
_, _, feature_dim = small_logits.shape
small_logits = torch.from_numpy(small_logits)
large_logits = torch.from_numpy(large_logits)

print(small_logits.shape, large_logits.shape)

small_logits = small_logits.reshape(-1, feature_dim)
large_logits = large_logits.reshape(-1, feature_dim)

print(small_logits.shape, large_logits.shape)

print(distillation(small_logits, large_logits, temp=5.0, alpha=0.7))
