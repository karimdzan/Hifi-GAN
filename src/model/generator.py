import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from typing import List

from src.config import GeneratorConfig

def dilation2padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2

def init_weights(module, mean=0., std=1e-2):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)

class ResBlock(torch.nn.Module):
    def __init__(self, n_channels: int, dilations: List[List[int]], kernel_size: int, leaky_relu_slope: float):
        super().__init__()
        self.conv_blocks = torch.nn.ModuleList()
        self.leaky_relu_slope = leaky_relu_slope
        for continual_dilations in dilations:
            layers = torch.nn.ModuleList()
            for dilation in continual_dilations:
                layers.append(
                    weight_norm(
                        torch.nn.Conv1d(
                            in_channels=n_channels, 
                            out_channels=n_channels, 
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=dilation2padding(kernel_size, dilation)
                    ))
                )
            self.conv_blocks.append(layers)
        self.conv_blocks.apply(init_weights)

    def forward(self, x):
        for block in self.blocks:
            skip = x
            for conv in block:
                skip = F.leaky_relu(skip, negative_slope=self.leaky_relu_slope, inplace=False)
                skip = conv(skip)
            x += skip
        return x

    def remove_weight_norm(self):
        for layers in self.blocks:
          for l in layers:
              remove_weight_norm(l)


class MRFLayer(torch.nn.Module):
    def __init__(self, config: GeneratorConfig, n_channels: int):
        super().__init__()
        self.config = config 
        self.resblocks = torch.nn.ModuleList()
        for kernel_size, dilations in zip(config.mrf_kernel_sizes, config.mrf_dilations):
            self.resblocks.append(
                ResBlock(
                    n_channels=n_channels,
                    dilations=dilations,
                    kernel_size=kernel_size,
                    lrelu_slope=config.leaky_relu_slope
                )
            )

    def forward(self, x):
        result = 0
        for block in self.resblocks:
            result += block(x)
        return result / len(self.resblocks)
    
    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()


class Generator(torch.nn.Module):
    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config
        self.blocks = torch.nn.Sequential()

        self.pre_conv = weight_norm(
            torch.nn.Conv1d(
                in_channels=config.mel_dimension,
                out_channels=config.upsampling_hidden_dim,
                kernel_size=config.pre_post_kernel_size,
                padding=config.pre_post_kernel_size // 2
        ))
        
        hidden_dim = config.upsampling_hidden_dim
        for kernel_size, stride in zip(config.upsampling_kernels, config.upsampling_strides):
            self.blocks.append(
                    weight_norm(
                        torch.nn.ConvTranspose1d(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim // 2,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size - stride) // 2
                    )))
            self.blocks.append(MRFLayer(config, hidden_dim // 2))
            hidden_dim //= 2

        self.post_conv = weight_norm(
            torch.nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=config.pre_post_kernel_size,
                padding=config.pre_post_kernel_size // 2
        ))
        self.pre_conv.apply(init_weights)
        self.post_conv.apply(init_weights)
        self.blocks.apply(init_weights)

    def count_params(self):
        gen_count = sum(p.numel() for p in self.parameters())
        print("Total gen parameters:  ", gen_count)
        return gen_count

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.blocks(x)
        x = self.post_conv(x).squeeze(1)
        return torch.tanh(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.pre_conv)
        remove_weight_norm(self.post_conv)
        for l in self.blocks:
            if isinstance(l, MRFLayer):
              l.remove_weight_norm()
            else:
              remove_weight_norm(l)
