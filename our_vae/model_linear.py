import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions



class ResBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x) + x


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, device, weight_dim=36737):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims, device=device, weight_dim=weight_dim)
        self.decoder = Decoder(latent_dims, weight_dim=weight_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class Encoder(nn.Module):
    def __init__(self, latent_dim, device, weight_dim, net_architect=[16385, 4096]):
        super(Encoder, self).__init__()
        net = []
        net_architect.insert(0, weight_dim) # add the input dim at the front
        for i in range(1, len(net_architect)):
            net.append(nn.Linear(net_architect[i-1], net_architect[i]))
            if i+1 != len(net_architect):
                net.append(nn.ReLU())
        self.network = nn.Sequential(*net)
        self.mu_network = nn.Linear(net_architect[-1], latent_dim)
        self.sigma_network = nn.Linear(net_architect[-1], latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.network(x)
        mu =  self.mu_network(x)
        sigma = torch.exp(self.sigma_network(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, weight_dim, net_architect=[4096, 16385]):
        super(Decoder, self).__init__()
        self.weight_dim = weight_dim
        net = []
        net_architect.insert(0, latent_dims) # add latend dim in the front
        net_architect.append(weight_dim) # add weight dim at the end
        for i in range(1, len(net_architect)):
            net.append(nn.Linear(net_architect[i-1], net_architect[i]))
            if i+1 != len(net_architect):
                net.append(nn.ReLU())
        self.network = nn.Sequential(*net)

    def forward(self, z):
        z = torch.sigmoid(self.network(z))
        return z.reshape((-1, self.weight_dim))



'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 16385]     601,952,130
              ReLU-2                [-1, 16385]               0
            Linear-3                 [-1, 4096]      67,117,056
            Linear-4                  [-1, 512]       2,097,664
            Linear-5                  [-1, 512]       2,097,664
           Encoder-6                  [-1, 512]               0
            Linear-7                 [-1, 4096]       2,101,248
              ReLU-8                 [-1, 4096]               0
            Linear-9                [-1, 16385]      67,129,345
             ReLU-10                [-1, 16385]               0
           Linear-11                [-1, 36737]     601,972,482
          Decoder-12                [-1, 36737]               0
================================================================
Total params: 1,344,467,589
Trainable params: 1,344,467,589
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.14
Forward/backward pass size (MB): 1.17
Params size (MB): 5128.74
Estimated Total Size (MB): 5130.04
----------------------------------------------------------------
'''