import torch.nn as nn
import torch
from chimaera.lib.models import DNAEncoder
from chimaera.lib.trainer import ChimaeraModule
import numpy as np

class OneToTwo(nn.Module):
    def __init__(self):
        super(OneToTwo, self).__init__()

    def forward(self, oned):
        twod1 = torch.tile(oned, [1, 1, 128])
        twod1 = torch.reshape(twod1, [-1, 64, 128, 128])
        twod2 = torch.permute(twod1, [0,1,3,2])
        twod  = torch.stack([twod1, twod2], dim=-1)
        twod = torch.mean(twod, dim=-1)
        return twod

class DNAEncoder_for_benchmark(DNAEncoder):
    def __init__(self, input_shape, latent_dim, head, **kwargs):
        super(DNAEncoder_for_benchmark, self).__init__(input_shape, latent_dim, n_outputs=1, **kwargs)
        self.head = head

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.downsampling_block(x)
        x = self.conv_block2(x)
        if self.residual:
            x = self.res_block(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        if self.head == 'chimaera':
            x = torch.flatten(x, 1)
            x = self.final_dropout(x)
            outputs = [self.fc[0](x)] # not using multy-output for benchmarking
        else:
            outputs = [x]
        return outputs

class AkitaHead(nn.Module):
    def __init__(self):
        super(AkitaHead, self).__init__()
        self.input_shape = (64, 128)
        self.one_to_two = OneToTwo()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=(2,1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(2,1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        r = 1
        l = []
        for i in range(6):
            l.append(ResBlock_akita_head(int(np.round(r))))
            r *= 1.75
        self.res_block = nn.Sequential(*l)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.one_to_two(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.res_block(x)
        x = self.last_conv(x)
        return x

class ResBlock_akita_head(nn.Module):
    def __init__(self, dilation_rate):
        super(ResBlock_akita_head, self).__init__()
        self.relu = nn.ReLU()
        padding = (dilation_rate * 2) // 2
        self.conv1 = nn.Conv2d(64, 32,
                               kernel_size=3,
                               padding=padding,
                               dilation=dilation_rate,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)


    def forward(self, x):
        recent_input = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += recent_input
        out = self.relu(out)
        return  out