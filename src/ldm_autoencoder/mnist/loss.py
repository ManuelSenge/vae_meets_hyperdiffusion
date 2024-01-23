import torch.nn as nn
import torch

class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()

    def forward(self, x, x_pred):
        return ((x - x_pred)**2).mean()