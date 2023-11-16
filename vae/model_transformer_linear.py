import torch
import torch.nn as nn
import torch.nn.functional as F
        
class SelfAttention(nn.Module):
    # heads needs to be 1 as the embedding dim is 1 and the embed size needs to be devisible by heads
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        #self.softmax = nn.Softmax(dim=2)
        self.multihead_attn = nn.MultiheadAttention(1, 1, batch_first=True)
        
    def forward(self, x):
        # x needs to be of shape (batch_size, seq_length, input_dim)
        print('attention in', x.shape)
        queries = self.query(x)
        print('queries', queries.shape)
        keys = self.key(x)
        print('keys', keys.shape)
        values = self.value(x)
        print('values', values.shape)
        attn_output, attn_output_weights = self.multihead_attn(queries, keys, values)
        print('attn_output', attn_output.shape)
        return attn_output
        '''print('keys.transpose', keys.transpose(1, 2).shape)
        print('torch.bmm(queries, keys.transpose(1, 2))', torch.bmm(queries, keys.transpose(1, 2)).shape)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        print('scores', scores.shape)
        attention = self.softmax(scores)
        print('attention', attention.shape)
        weighted = torch.bmm(attention, values)
        print('weighted', weighted.shape)
        return weighted # shape (batch_size, seq_length, input_dim)'''

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, device, att_layers, net_architect):
        super(Encoder, self).__init__()

        # self-attention layers
        att_network = []
        for _ in range(att_layers):
            att_network.append(SelfAttention(1))
        self.att_network = nn.Sequential(*att_network)

        # FF Encoder network
        net = []
        net_architect.insert(0, input_dim) # add the input dim at the front
        for i in range(1, len(net_architect)):
            net.append(nn.Linear(net_architect[i-1], net_architect[i]))
            if i+1 != len(net_architect):
                net.append(nn.ReLU())
        self.network = nn.Sequential(*net)

        # latend space parameters
        self.mu_network = nn.Linear(net_architect[-1], latent_dim)
        self.sigma_network = nn.Linear(net_architect[-1], latent_dim)

        # sampling
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        # x is in shape (BS, seq_len) but att_network expects it in (batch_size, seq_length, input_dim)
        print('x', x.shape)
        x = x.view(x.shape[0], x.shape[1], 1) #1->2
        print('x', x.shape)
        #x = x.view(x.shape[0], x.shape[1], 1)
        x = self.att_network(x)
        print('attented_x', x.shape)
        x = torch.flatten(x, start_dim=1)
        print('attent_x', x.shape)
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


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dims, device, att_layers=1, net_architect=[16385, 4096]):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dims, device, att_layers=att_layers, net_architect=net_architect)
        self.decoder = Decoder(latent_dims, weight_dim=input_dim)

    def forward(self, x):
        # encoder expects in as (BS, embed dim (1), seq) got (BS, seq)
        #x = x.view(x.shape[0], 1, x.shape[1])
        z = self.encoder(x)
        print('z', z.shape)
        return self.decoder(z)

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1             [-1, 36737, 1]               2
            Linear-2             [-1, 36737, 1]               2
            Linear-3             [-1, 36737, 1]               2
           Softmax-4         [-1, 36737, 36737]               0
     SelfAttention-5             [-1, 36737, 1]               0
            Linear-6             [-1, 36737, 1]               2
            Linear-7             [-1, 36737, 1]               2
            Linear-8             [-1, 36737, 1]               2
           Softmax-9         [-1, 36737, 36737]               0
    SelfAttention-10             [-1, 36737, 1]               0
           Linear-11                [-1, 16385]     601,952,130
             ReLU-12                [-1, 16385]               0
           Linear-13                 [-1, 4096]      67,117,056
           Linear-14                  [-1, 512]       2,097,664
           Linear-15                  [-1, 512]       2,097,664
          Encoder-16                  [-1, 512]               0
           Linear-17                 [-1, 4096]       2,101,248
             ReLU-18                 [-1, 4096]               0
           Linear-19                [-1, 16385]      67,129,345
             ReLU-20                [-1, 16385]               0
           Linear-21                [-1, 36737]     601,972,482
          Decoder-22                [-1, 36737]               0
================================================================
Total params: 1,344,467,601
Trainable params: 1,344,467,601
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.14
Forward/backward pass size (MB): 20596.78
Params size (MB): 5128.74
Estimated Total Size (MB): 25725.66
----------------------------------------------------------------
'''