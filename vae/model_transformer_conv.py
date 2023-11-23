import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, weight_dim, channels=1, enc_chans=[64, 32, 16, 1], latent_dim=512, act_fn=nn.ReLU(), enc_kernal_sizes=[8, 6, 3, 3], self_attention=[0,0,0,0], device=torch.device('cpu')):
        super(Encoder, self).__init__()
        self.enc_chans = enc_chans
        self.dec_chans = list(reversed(self.enc_chans))
        self.activ_func = act_fn
        self.weight_dim = weight_dim
        self.self_attention = self_attention
        dims = [36737, 18365, 9180, 4589, 2294]
        
        
        self.enc_chans = [channels, *enc_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (self.enc_chans, self.dec_chans))
        self.enc_conv_layers = []

        self.pooling = []
        self.transformer_encoder = []
        # add some conv layers
        for (enc_in, enc_out), enc_kernel, d, att_flag in zip(enc_chans_io, enc_kernal_sizes, dims, self_attention):
            self.enc_conv_layers.append(nn.Conv1d(in_channels=enc_in, out_channels=enc_out, kernel_size=enc_kernel, stride = 2, padding = 1))
            self.pooling.append(nn.MaxPool1d(2, stride=1, return_indices=True))
            if bool(att_flag):
                encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=8)
                self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
            else:
                self.transformer_encoder.append(None)
        
        # add all layers to module list
        self.enc_conv_layers = nn.ModuleList(self.enc_conv_layers)
        self.pooling = nn.ModuleList(self.pooling)

        linear_input_dim = 2294
        # latend space parameters
        self.mu_network = nn.Linear(linear_input_dim, latent_dim)
        self.sigma_network = nn.Linear(linear_input_dim, latent_dim)
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        # sampling
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        output = x.view(-1, 1, self.weight_dim) # add one channel
        indices_pooling = []
        output_sizes = []
        # encoder
        for i in range(len(self.pooling)):
            if not self.transformer_encoder[i] is None:
                output = self.transformer_encoder[i](output)
            output = self.enc_conv_layers[i](output)  # torch.Size([BS, FILTER(8, 16, 32), height(499, 122, 30), width(499, 122, 3030)])
            output_sizes.append(output.shape)
            output, indx = self.pooling[i](output)  # torch.Size([BS, FILTER(8, 16, 32), height(246, 59, 14), width(246, 59, 14)])
            indices_pooling.append(indx)
            output = self.activ_func(output)
        

        mu =  self.mu_network(output)
        sigma = torch.exp(self.sigma_network(output))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, indices_pooling, output_sizes

class Decoder(nn.Module):
    def __init__(self, weight_dim, channels=1, enc_chans=[64, 32, 16, 1], latent_dim=512, act_fn=nn.ReLU(), enc_kernal_sizes=[8, 6, 3, 3], self_attention=[0,0,0,0], device=torch.device('cpu')):
        super(Decoder, self).__init__()
        self.activ_func = act_fn
        self.weight_dim = weight_dim
        enc_chans = [channels, *enc_chans]
        self.dec_chans = list(reversed(enc_chans))
        dec_kernal_sizes = list(reversed(enc_kernal_sizes))
        self.self_attention = self_attention
        dims = [36737, 18365, 9180, 4589, 2294]
        dims.reverse()


        #dec_init_chan = self.dec_chans[0]
        #self.dec_chans = [dec_init_chan, *self.dec_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, self.dec_chans))
        self.dec_conv_layers = []

        self.unpooling = []
        output_padding = [0,1,1,1]
        
        self.transformer_encoder = []
        # add some conv layers
        for (dec_in, dec_out), dec_kernel, out_p, d, att_flag in zip(dec_chans_io, dec_kernal_sizes, output_padding, dims, self_attention):
            self.dec_conv_layers.append(nn.ConvTranspose1d(in_channels=dec_in, out_channels=dec_out, kernel_size=dec_kernel, stride = 2, padding = 1, output_padding=out_p))
            self.unpooling.append(nn.MaxUnpool1d(2, stride=1))
            if bool(att_flag):
                encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=1)
                self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
            else:
                self.transformer_encoder.append(None)
        
        # add all layers to module list
        self.dec_conv_layers = nn.ModuleList(self.dec_conv_layers)
        self.unpooling = nn.ModuleList(self.unpooling)

        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.linear_input_output_dim = 2294
        self.linear = nn.Sequential(nn.Linear(latent_dim, self.linear_input_output_dim),act_fn)

    def forward(self, z, indices_pooling, output_sizes):
        indices_pooling.reverse()
        output_sizes.reverse()
        output = self.linear(z)
        for i in range(len(self.dec_conv_layers)):
            if not self.transformer_encoder[i] is None:
                output = self.transformer_encoder[i](output)
            output = self.unpooling[i](output, indices=indices_pooling[i], output_size=output_sizes[i]) #  torch.Size([BS, FILTER(32, 16, 8), height(30, 122, 499), height(30, 122, 499)])
            output = self.dec_conv_layers[i](output)  # torch.Size([BS, FILTER(32, 16, 8), height(59, 122, 499), width(59, 122, 499)])
            output = self.activ_func(output)

        #z = torch.sigmoid(self.network(z))
        return output.reshape((-1, self.weight_dim))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dims, device, enc_chans=[64, 32, 16, 1], enc_kernal_sizes=[8, 6, 3, 3], self_attention_encoder=[0,0,0,0], self_attention_decoder=[0,0,0,0]):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dims, weight_dim=input_dim, enc_chans=enc_chans, enc_kernal_sizes=enc_kernal_sizes, device=device, self_attention=self_attention_encoder)
        self.decoder = Decoder(latent_dim=latent_dims, weight_dim=input_dim, enc_kernal_sizes=enc_kernal_sizes, enc_chans=enc_chans, device=device, self_attention=self_attention_decoder)

    def forward(self, x):
        # encoder expects in as (BS, embed dim (1), seq) got (BS, seq)
        #x = x.view(x.shape[0], 1, x.shape[1])
        z, indices_pooling, output_sizes = self.encoder(x)
        return self.decoder(z, indices_pooling, output_sizes)

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