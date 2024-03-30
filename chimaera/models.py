# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class EnsembleModel(nn.Module):
    '''Combine two pytorch models'''
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.input_shape = modelA.input_shape # for torchsummary

    def forward(self, x):
        outputs = self.modelA(x)
        outputs = [self.modelB(output) for output in outputs]
        return torch.cat(outputs, dim=1)

class VAE(nn.Module):
    '''Variational autoencoder'''
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = encoder.input_shape # for torchsummary

    def forward(self, x):
        z_mean, z_log_var, z = self.encoder.forward_vae_mode(x)
        outputs = self.decoder(z)
        return outputs, z_mean, z_log_var, z

class HiCEncoder(nn.Module):
    def __init__(
            self,
            input_shape,
            latent_dim,
            ):
        super(HiCEncoder, self).__init__()
        self.h, self.w, self.channels = input_shape
        self.input_shape = (self.channels, self.h, self.w) # for torchsummary
        self.output_shape = (latent_dim, )
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.z_mean = nn.Linear(self.h // 8 * self.w // 8 * 64, latent_dim)
        self.z_log_var = nn.Linear(self.h // 8 * self.w // 8 * 64, latent_dim)
        self.N = torch.distributions.Normal(0, 0.1)
        if torch.cuda.is_available():
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def _forward_first_layers(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        return x

    def forward_simple_mode(self, x):
        x = self._forward_first_layers(x)
        x = self.z_mean(x)
        return x

    def forward_vae_mode(self, x):
        x = self._forward_first_layers(x)
        mu =  self.z_mean(x)
        log_sigma = self.z_log_var(x)
        z = mu + torch.exp(log_sigma) * self.N.sample(mu.shape)
        return mu, log_sigma, z

    def forward(self, x):
        return self.forward_simple_mode(x)

class HiCDecoder(nn.Module):
    def __init__(
            self,
            output_shape,
            latent_dim,
            ):
        super(HiCDecoder, self).__init__()
        self.input_shape = (latent_dim,) # for torchsummary
        self.h, self.w, self.channels = output_shape
        self.latent_dim = latent_dim
        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(latent_dim, self.h // 8 * self.w // 8 * 64)
        self.conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=0, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=0, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=0, stride=2, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.ConvTranspose2d(16, self.channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = torch.reshape(x, (-1, 64, (self.h // 8), (self.w // 8)))
        x = self.conv1(x)[:, :, :-1, :-1]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)[:, :, :-1, :-1]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)[:, :, :-1, :-1]
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        return x

class ResBlock(nn.Module):
    def __init__(self,
            filters,
            kernel_sizes,
            dropout_rates,
            dilation_rates,
            extra_layer=False,
            stride=1,
            track_running_stats=False
        ):
        super(ResBlock, self).__init__()
        self.extra_layer = extra_layer
        if self.extra_layer:
            self.conv0 = nn.Conv1d(filters[0], filters[2],
                               kernel_size=1,
                               padding=0,
                               stride=stride)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rates[0])
        padding1 = (dilation_rates[0] * (kernel_sizes[0] - 1)) // 2
        self.conv1 = nn.Conv1d(filters[0], filters[1],
                               kernel_size=kernel_sizes[0],
                               padding=padding1,
                               dilation=dilation_rates[0],
                               bias=False)
        self.bn1 = nn.BatchNorm1d(filters[1],
                                     track_running_stats=track_running_stats)

        self.dropout2 = nn.Dropout(dropout_rates[1])
        padding2 = (dilation_rates[1] * (kernel_sizes[1] - 1)) // 2
        self.conv2 = nn.Conv1d(filters[1], filters[2],
                               kernel_size=kernel_sizes[1],
                               stride=stride,
                               padding=padding2,
                               dilation=dilation_rates[1],
                               bias=False)
        self.bn2 = nn.BatchNorm1d(filters[2],
                                     track_running_stats=track_running_stats)


    def forward(self, x):
        recent_input = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        if not self.extra_layer:
            out += recent_input
        else:
            if hasattr(self, 'conv0'):
                out = out + self.conv0(recent_input)
            else:
                out += recent_input
        out = self.relu(out)
        return  out

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dropout_rate,
            dilation_rate,
            downsample=False,
            track_running_stats=False):
        super(ConvBlock, self).__init__()
        self.downsample = downsample
        self.max_pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        padding = (dilation_rate * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation_rate)
        self.bn = nn.BatchNorm1d(
            out_channels,
            track_running_stats=track_running_stats
            )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.downsample:
            x = self.max_pool(x)
        x = self.dropout(x)
        return x

class DNAEncoder(nn.Module):
    def __init__(
            self,
            input_shape,
            latent_dim,
            n_outputs,
            filters = [64, 128, 256, 512, 64],
            kernel_sizes = [15, 9, 5, 3, 3],
            dilation_rates = [1, 2, 2, 1, 1],
            all_conv_residual = False,
            n_residual_blocks = 8,
            residual_block_filters = 256,
            residual_block_kernel_sizes = [3, 3],
            residual_block_extra_layer = False,
            residual_block_dilation_rate_increasing = 'linear',
            residual_block_dilation_rate_factors = [2, 2],
            dropout_rates = [0, 0, 0, 0, 0.15],
            residual_block_dropout_rates = [0, 0],
            track_running_stats=False
            ):
        super(DNAEncoder, self).__init__()
        dropout_rates = [float(i) for i in dropout_rates]
        residual_block_dropout_rates = [float(i) for i in  residual_block_dropout_rates]
        self.input_shape = (input_shape[1], input_shape[0]) # for torchsummary
        self.latent_dim = latent_dim
        self.residual = n_residual_blocks > 0
        self.conv_block1 = ConvBlock(4, filters[0], kernel_size=kernel_sizes[0],
                                     dropout_rate=dropout_rates[0],
                                     dilation_rate=dilation_rates[0],
                                     downsample=True,
                                     track_running_stats=track_running_stats)
        n_poolings = int(np.log2(input_shape[0])) - 10
        resulting_len = int(input_shape[0] / 2 ** (n_poolings + 3)) # should be 128 if input length is a power of 2
        l = []
        if not all_conv_residual:
            for i in range(n_poolings):
                out_channels = filters[1]
                in_channels = filters[1] if i else filters[0]
                l.append(ConvBlock(in_channels, out_channels,
                                kernel_size=kernel_sizes[1],
                                dropout_rate=dropout_rates[1],
                                dilation_rate=dilation_rates[1],
                                downsample=True,
                                     track_running_stats=track_running_stats))
        else:
            for i in range(n_poolings):
                out_channels = filters[1]
                in_channels = filters[1] if i else filters[0]
                l.append(ResBlock(filters=[in_channels, filters[1], out_channels],
                                kernel_sizes=[kernel_sizes[1], kernel_sizes[1]],
                                dropout_rates=[dropout_rates[1],dropout_rates[1]],
                                dilation_rates=[dilation_rates[1],dilation_rates[1]],
                                extra_layer=True,
                                stride=2,
                                     track_running_stats=track_running_stats))

        self.downsampling_block = nn.Sequential(*l)
        self.conv_block2 = ConvBlock(filters[1], filters[1],
                                     kernel_size=kernel_sizes[1],
                                     dropout_rate=0.0,
                                     dilation_rate=dilation_rates[1],
                                     downsample=False,
                                     track_running_stats=track_running_stats)
        d1, d2 = residual_block_dilation_rate_factors
        l = []
        for i in range(n_residual_blocks):
            if residual_block_dilation_rate_increasing == 'linear':
                residual_block_dilation_rates = [d1*(i+1), d2*(i+1)]
            else:
                residual_block_dilation_rates = [d1**i, d2**i]
            l.append(ResBlock(
            filters = [filters[1], residual_block_filters, filters[1]],
            kernel_sizes = residual_block_kernel_sizes,
            dropout_rates = residual_block_dropout_rates,
            extra_layer = residual_block_extra_layer,
            dilation_rates = residual_block_dilation_rates,
            track_running_stats=track_running_stats
            )
            )
        self.res_block = nn.Sequential(*l)
        self.conv_block3 = ConvBlock(filters[1], filters[2],
                                     kernel_size=kernel_sizes[2],
                                     dropout_rate=dropout_rates[2],
                                     dilation_rate=dilation_rates[2],
                                     downsample=False,
                                     track_running_stats=track_running_stats)
        self.conv_block4 = ConvBlock(filters[2], filters[3],
                                     kernel_size=kernel_sizes[3],
                                     dropout_rate=dropout_rates[3],
                                     dilation_rate=dilation_rates[3],
                                     downsample=True,
                                     track_running_stats=track_running_stats)
        self.conv_block5 = ConvBlock(filters[3], filters[4],
                                     kernel_size=kernel_sizes[4],
                                     dropout_rate=0.0,
                                     dilation_rate=dilation_rates[4],
                                     downsample=True,
                                     track_running_stats=track_running_stats)
        self.fc = nn.ModuleList(
            [nn.Linear(filters[4]*resulting_len, latent_dim) for i in range(n_outputs)]
        )
        self.final_dropout = nn.Dropout(dropout_rates[4])

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.downsampling_block(x)
        x = self.conv_block2(x)
        if self.residual:
            x = self.res_block(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = torch.flatten(x, 1)
        x = self.final_dropout(x)
        outputs = [fc(x) for fc in self.fc]
        return outputs


def hic_encoder(
        input_shape=(32,128,1),
        latent_dim=96,
        ):
    model = HiCEncoder(input_shape, latent_dim)
    model.apply(init_weights)
    return model

def hic_decoder(
        output_shape=(32,128,1),
        latent_dim=96,
        ):
    model = HiCDecoder(output_shape, latent_dim)
    model.apply(init_weights)
    return model

def dna_encoder(
    input_shape,
    latent_dim,
    **kwargs
    ):
    model = DNAEncoder(input_shape, latent_dim, **kwargs)
    model.apply(init_weights)
    return model

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)