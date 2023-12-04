from dataclasses import dataclass, field
import torch
from typing import List, Tuple


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


@dataclass
class GeneratorConfig:
    upsampling_strides: Tuple[int] = (8, 8, 2, 2)
    upsampling_kernels: Tuple[int] = (16, 16, 4, 4)
    upsampling_hidden_dim: int = 512

    mrf_dilations: List[List[List[int]]] = field(default_factory=lambda: [[[1, 1], [3, 1], [5, 1]]] * 3)
    mrf_kernel_sizes: Tuple[int] = (3, 7, 11)

    pre_post_kernel_size: int = 7

    mel_dimension: int = 80

    leaky_relu_slope: float = 0.1


@dataclass
class DiscriminatorConfig:
    leaky_relu_slope: float = 0.1
    mpd_kernel_size: int = 5
    mpd_stride: int = 3
    mpd_n_blocks: int = 4
    mpd_post_conv_kernel_size: int = 3
    mpd_periods: Tuple[int] = (2, 3, 5, 7, 11)

    msd_channels: Tuple[int] = (1, 128, 128, 256, 512, 1024, 1024, 1024, 1)
    msd_kernel_sizes: Tuple[int] = (15, 41, 41, 41, 41, 41, 5, 3)
    msd_strides: Tuple[int] = (1, 2, 2, 4, 4, 1, 1, 1)
    msd_paddings: Tuple[int] = (7, 20, 20, 20, 20, 20, 2, 1)
    msd_groups: Tuple[int] = (1, 4, 16, 16, 16, 16, 1, 1)

    pool_kernel: int = 4
    pool_stride: int = 2
    pool_padding: int = 2


@dataclass
class TrainConfig:
    train_path: str = "./data/LJSpeech-1.1"
    wav_path: str = "./data/test"
    logs_path: str = "./results"

    frame_len: int = 8192 * 2 #remove *2 if it fails
    device: str = "cuda"
    seed: int = 42

    adam_betas: Tuple[float] = (0.8, 0.99)
    lr_decay: float = 0.999
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    batch_size: float = 32

    l1_gamma: float = 45.
    gan_gamma: float = 1.
    fm_gamma: float = 2.

    epochs: int = 500
    log_steps: int = 10
    save_steps: int = 1000

    wandb_project: str = "HiFi-GAN"

    max_wav_value: float = 32768.0
    sample_rate: int = 22050


mel_config = MelSpectrogramConfig()
generator_config = GeneratorConfig()
discriminator_config = DiscriminatorConfig()
train_config = TrainConfig()