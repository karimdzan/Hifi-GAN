import torch
import torch.nn.functional as F
import random
import torchaudio
import os

from src.config import TrainConfig, MelSpectrogramConfig
from src.mel import MelSpectrogram


class MelWavDataset(torch.utils.data.Dataset):
    def __init__(self, train_config: TrainConfig, mel_config: MelSpectrogramConfig):
        self.config = train_config
        self.mel_config = mel_config
        self.melspec = MelSpectrogram(mel_config, center=False).to(train_config.device)
        self.train_wavs = os.listdir(os.path.join(train_config.train_path, "wavs"))

    def __len__(self):
        return len(self.train_wavs)

    def melspec_with_pad(self, wav):
        pad_size = self.mel_config.n_fft - self.mel_config.hop_length
        wav_padded = F.pad(wav, (pad_size // 2, pad_size // 2), mode='reflect')
        return self.melspec(wav_padded)
    
    def __getitem__(self, idx):
        wav_file = os.path.join(self.config.train_path, "wavs", self.train_wavs[idx])
        wav_data = torchaudio.load(wav_file)[0].to(self.config.device)
        # wav_data = wav_data / self.config.max_wav_value
        if wav_data.size(1) < self.config.frame_len:
            wav_data = F.pad(wav_data, (0, self.config.frame_len - wav_data.size(1)))
        elif wav_data.size(1) > self.config.frame_len:
            start = random.randint(0, wav_data.size(1) - self.config.frame_len)
            wav_data = wav_data[:, start: start + self.config.frame_len]

        mel = self.melspec_with_pad(wav_data)
        return wav_data, mel