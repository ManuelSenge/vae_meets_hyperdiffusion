import torch.nn as nn


class VAELoss(nn.Module):
    def __init__(self, autoencoder):
        super(VAELoss, self).__init__()
        self.autoencoder = autoencoder

    def forward(self, x, x_pred):
        return ((x - x_pred)**2).sum() + self.autoencoder.encoder.kl