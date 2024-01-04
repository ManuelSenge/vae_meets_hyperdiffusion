import torch.nn as nn
import torch

class VAELoss(nn.Module):
    def __init__(self, autoencoder):
        super(VAELoss, self).__init__()
        self.autoencoder = autoencoder

    def forward(self, x, x_pred, variational, posterior=None):
        if variational:
            return ((x - x_pred)**2).mean(), torch.mean(posterior.kl())
        else:
            return ((x - x_pred)**2).mean(), torch.Tensor([0])