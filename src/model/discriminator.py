import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from src.config import DiscriminatorConfig

def dilation2padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2

class PeriodBlock(torch.nn.Module):
    def __init__(self, period, config: DiscriminatorConfig):
        super().__init__()
        self.period = period
        self.config = config
        self.conv_blocks = torch.nn.ModuleList()
        current_in_dim = 1
        for i in range(1, config.mpd_n_blocks * 2, 2):
            if i == (config.mpd_n_blocks * 2 - 1):
                current_out_dim = 2 ** (i + 3)
            else:
                current_out_dim = 2 ** (i + 4)
            self.conv_blocks.append(weight_norm(nn.Conv2d(
                in_channels = current_in_dim,
                out_channels = current_out_dim, 
                kernel_size=(config.mpd_kernel_size, 1), 
                stride=(config.mpd_stride, 1), 
                padding=(dilation2padding(config.mpd_kernel_size, 1), 0)
                    )   
                )
            )
            current_in_dim = current_out_dim
        self.conv_blocks.append(weight_norm(nn.Conv2d(in_channels=current_in_dim, 
                                                      out_channels=current_in_dim, 
                                                      kernel_size=(config.mpd_kernel_size, 1), 
                                                      stride=1, 
                                                      padding=(2, 0)
                                                      )
                                                    )
                                                  )
        self.conv_post = weight_norm(nn.Conv2d(in_channels=current_in_dim, 
                                               out_channels=1, 
                                               kernel_size=(config.mpd_post_conv_kernel_size, 1), 
                                               stride=1, 
                                               padding=(1, 0)))

    def forward(self, x, fmap):
        tmp = []
        x = x.unsqueeze(1)
        bsz, ch, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(bsz, ch, t // self.period, self.period).contiguous()

        for conv in self.conv_blocks:
            x = conv(x)
            x = F.leaky_relu(x, self.config.leaky_relu_slope)
            tmp.append(x)

        x = self.conv_post(x)
        tmp.append(x)
        fmap.append(tmp)
        return x.flatten(1, -1)


class ScaleBlock(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig, conv_norm=weight_norm):
        super().__init__()
        self.config = config

        self.leaky_relu = torch.nn.LeakyReLU(config.leaky_relu_slope)
        self.conv_blocks = torch.nn.ModuleList()

        for in_chans, out_chans, kernel, stride, padding, groups in zip(
            config.msd_channels[:-1], 
            config.msd_channels[1:],
            config.msd_kernel_sizes, 
            config.msd_strides,
            config.msd_paddings, 
            config.msd_groups
        ):
            self.conv_blocks.append(
                conv_norm(torch.nn.Conv1d(
                    in_channels=in_chans,
                    out_channels=out_chans,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    groups=groups
                ))
            )

    def forward(self, x, fmap):
        tmp = []
        for i, conv in enumerate(self.conv_blocks):
            x = conv(x)
            if i != (len(self.conv_blocks) - 1):
                x = self.leaky_relu(x)
            tmp.append(x)
        fmap.append(tmp)
        return x.flatten(1, -1)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in config.mpd_periods:
            self.discriminators.append(PeriodBlock(period, config))

    def count_params(self):
        print("MPD param count: ", sum(p.numel() for p in self.parameters()))

    def forward(self, y, y_hat):
        y_rs, y_gs = [], []
        fmap_rs, fmap_gs = [], []
        for d in self.discriminators:
            y_r = d(y, fmap_rs)
            y_g = d(y_hat, fmap_gs)
            y_rs.append(y_r)
            y_gs.append(y_g)
        return y_rs, y_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleBlock(config, spectral_norm),
            ScaleBlock(config, weight_norm),
            ScaleBlock(config, weight_norm),
        ])
        self.mean_pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def count_params(self):
        print("MSD param count: ", sum(p.numel() for p in self.parameters()))

    def forward(self, y, y_hat):
        fmap_rs, fmap_gs = [], []
        y = y.unsqueeze(1)
        y_hat = y_hat.unsqueeze(1)
        y_r = self.discriminators[0](y, fmap_rs)
        y_g = self.discriminators[0](y_hat, fmap_gs)
        y_rs, y_gs = [y_r], [y_g]
        for i, pool in enumerate(self.mean_pools):
            y = pool(y)
            y_hat = pool(y_hat)
            y_r = self.discriminators[i + 1](y, fmap_rs)
            y_g = self.discriminators[i + 1](y_hat, fmap_gs)
            y_rs.append(y_r)
            y_gs.append(y_g)
        return y_rs, y_gs, fmap_rs, fmap_gs