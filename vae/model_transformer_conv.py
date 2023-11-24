import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, weight_dim, channels=1, enc_chans=[64, 32, 16, 1], latent_dim=512, act_fn=nn.ReLU(), enc_kernal_sizes=[8, 6, 3, 3], self_attention=[0,0,0,0], device=torch.device('cpu'), num_att_layers=4):
        super(Encoder, self).__init__()
        self.enc_chans = enc_chans
        self.dec_chans = list(reversed(self.enc_chans))
        self.activ_func = act_fn
        self.weight_dim = weight_dim
        self.self_attention = self_attention
        self.device = device
        dims = [36737, 18366, 9182, 4591, 2296]
        nheads = [1, 1, 2, 1, 1]
        
        
        self.enc_chans = [channels, *enc_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (self.enc_chans, self.dec_chans))
        self.enc_conv_layers = []

        self.batch_normalization = []
        self.transformer_encoder = []
        # add some conv layers
        for (enc_in, enc_out), enc_kernel, d, att_flag, nhead in zip(enc_chans_io, enc_kernal_sizes, dims, self_attention, nheads):
            self.enc_conv_layers.append(nn.Conv1d(in_channels=enc_in, out_channels=enc_out, kernel_size=enc_kernel, stride = 2, padding = 1))
            #self.pooling.append(nn.MaxPool1d(2, stride=1, return_indices=True))
            if bool(att_flag):
                encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead)
                self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=num_att_layers))
            else:
                self.transformer_encoder.append(None)
            self.batch_normalization.append(nn.BatchNorm1d(enc_out))

            
        
        # add all layers to module list
        self.enc_conv_layers = nn.ModuleList(self.enc_conv_layers)
        self.batch_normalization = nn.ModuleList(self.batch_normalization)
        #self.pooling = nn.ModuleList(self.pooling)

        linear_input_dim = 2296
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
        #indices_pooling = []
        #output_sizes = []
        # encoder
        for i in range(len(self.enc_conv_layers)):
            if not self.transformer_encoder[i] is None:
                output = self.transformer_encoder[i](output)
            output = self.enc_conv_layers[i](output)  # torch.Size([BS, FILTER(8, 16, 32), height(499, 122, 30), width(499, 122, 3030)])
            #output_sizes.append(output.shape)
            #output, indx = self.pooling[i](output)  # torch.Size([BS, FILTER(8, 16, 32), height(246, 59, 14), width(246, 59, 14)])
            #indices_pooling.append(indx)
            output = self.activ_func(output)
            output = self.batch_normalization[i](output)
        

        mu =  self.mu_network(output)
        sigma = torch.exp(self.sigma_network(output))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, weight_dim, channels=1, enc_chans=[64, 32, 16, 1], latent_dim=512, act_fn=nn.ReLU(), enc_kernal_sizes=[8, 6, 3, 3], self_attention=[0,0,0,0], device=torch.device('cpu'), num_att_layers=4):
        super(Decoder, self).__init__()
        self.activ_func = act_fn
        self.weight_dim = weight_dim
        enc_chans = [channels, *enc_chans]
        self.dec_chans = list(reversed(enc_chans))
        dec_kernal_sizes = list(reversed(enc_kernal_sizes))
        self.self_attention = self_attention
        dims = [36737, 18366, 9182, 4591, 2296]
        nheads = [1, 1, 2, 1, 1]
        nheads.reverse()
        dims.reverse()


        #dec_init_chan = self.dec_chans[0]
        #self.dec_chans = [dec_init_chan, *self.dec_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, self.dec_chans))
        self.dec_conv_layers = []

        self.batch_normalization = []
        output_padding = [0,1,0,1]
        
        self.transformer_encoder = []
        # add some conv layers
        for (dec_in, dec_out), dec_kernel, out_p, d, att_flag, nhead in zip(dec_chans_io, dec_kernal_sizes, output_padding, dims, self_attention, nheads):
            self.dec_conv_layers.append(nn.ConvTranspose1d(in_channels=dec_in, out_channels=dec_out, kernel_size=dec_kernel, stride = 2, padding = 1, output_padding=out_p))
            #self.unpooling.append(nn.MaxUnpool1d(2, stride=1))
            if bool(att_flag):
                encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead)
                self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=num_att_layers))
            else:
                self.transformer_encoder.append(None)

            self.batch_normalization.append(nn.BatchNorm1d(dec_out))
        
        # add all layers to module list
        self.dec_conv_layers = nn.ModuleList(self.dec_conv_layers)
        #self.unpooling = nn.ModuleList(self.unpooling)
        self.batch_normalization = nn.ModuleList(self.batch_normalization)

        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)

        self.linear_input_output_dim = 2296
        self.linear = nn.Sequential(nn.Linear(latent_dim, self.linear_input_output_dim),act_fn)

    def forward(self, z):
        #if not indices_pooling is None:
        #    indices_pooling.reverse()
        #output_sizes.reverse()
        output = self.linear(z)
        for i in range(len(self.dec_conv_layers)):
            if not self.transformer_encoder[i] is None:
                output = self.transformer_encoder[i](output)
            # indices=indices_pooling[i], 
            #output = self.unpooling[i](output, output_size=output_sizes[i]) #  torch.Size([BS, FILTER(32, 16, 8), height(30, 122, 499), height(30, 122, 499)])
            output = self.dec_conv_layers[i](output)  # torch.Size([BS, FILTER(32, 16, 8), height(59, 122, 499), width(59, 122, 499)])
            output = self.activ_func(output)
            output = self.batch_normalization[i](output)

        #z = torch.sigmoid(self.network(z))
        return output.reshape((-1, self.weight_dim))


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dims, device, enc_chans=[64, 32, 16, 1], enc_kernal_sizes=[8, 6, 3, 3], self_attention_encoder=[0,0,0,0], self_attention_decoder=[0,0,0,0], num_att_layers=4):
        super(VariationalAutoencoder, self).__init__(), 
        self.encoder = Encoder(latent_dim=latent_dims, weight_dim=input_dim, enc_chans=enc_chans, enc_kernal_sizes=enc_kernal_sizes, device=device, self_attention=self_attention_encoder, num_att_layers=num_att_layers)
        self.decoder = Decoder(latent_dim=latent_dims, weight_dim=input_dim, enc_kernal_sizes=enc_kernal_sizes, enc_chans=enc_chans, device=device, self_attention=self_attention_decoder, num_att_layers=num_att_layers)

    def forward(self, x):
        # encoder expects in as (BS, embed dim (1), seq) got (BS, seq)
        #x = x.view(x.shape[0], 1, x.shape[1])
        z = self.encoder(x)
        return self.decoder(z)

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1            [-1, 64, 18366]             576
              ReLU-2            [-1, 64, 18366]               0
            Conv1d-3             [-1, 32, 9182]          12,320
              ReLU-4             [-1, 32, 9182]               0
            Conv1d-5             [-1, 16, 4591]           1,552
              ReLU-6             [-1, 16, 4591]               0
            Conv1d-7              [-1, 1, 2296]              49
              ReLU-8              [-1, 1, 2296]               0
            Linear-9               [-1, 1, 512]       1,176,064
           Linear-10               [-1, 1, 512]       1,176,064
          Encoder-11               [-1, 1, 512]               0
           Linear-12              [-1, 1, 2296]       1,177,848
             ReLU-13              [-1, 1, 2296]               0
             ReLU-14              [-1, 1, 2296]               0
  ConvTranspose1d-15             [-1, 16, 4591]              64
             ReLU-16             [-1, 16, 4591]               0
             ReLU-17             [-1, 16, 4591]               0
  ConvTranspose1d-18             [-1, 32, 9182]           1,568
             ReLU-19             [-1, 32, 9182]               0
             ReLU-20             [-1, 32, 9182]               0
  ConvTranspose1d-21            [-1, 64, 18366]          12,352
             ReLU-22            [-1, 64, 18366]               0
             ReLU-23            [-1, 64, 18366]               0
  ConvTranspose1d-24             [-1, 1, 36737]             513
             ReLU-25             [-1, 1, 36737]               0
             ReLU-26             [-1, 1, 36737]               0
          Decoder-27                [-1, 36737]               0
================================================================
Total params: 3,558,970
Trainable params: 3,558,970
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.14
Forward/backward pass size (MB): 60.07
Params size (MB): 13.58
Estimated Total Size (MB): 73.79
----------------------------------------------------------------
'''