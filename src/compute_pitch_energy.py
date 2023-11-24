from src.waveglow.audio.stft import TacotronSTFT
from src.config import train_config, model_config, mel_config
from src.data import process_text
from tqdm import tqdm
import numpy as np
import torch
import librosa
import os
import pyworld
import src.waveglow.audio.hparams_audio as audio_params


def compute_energy(wav):
    STFT = TacotronSTFT(filter_length=model_config.filter_length, 
                        hop_length=audio_params.hop_length, 
                        win_length=model_config.win_length, 
                        n_mel_channels=mel_config.num_mels, 
                        sampling_rate=audio_params.sampling_rate, 
                        mel_fmin=0, 
                        mel_fmax=8000)

    wav = torch.FloatTensor(wav)
    wav = torch.clip(wav.unsqueeze(0), -1, 1)

    audiograd = torch.autograd.Variable(wav, requires_grad=False)
    energy = torch.norm(STFT.mel_spectrogram(audiograd)[1], dim=1)

    energy = energy.squeeze(0).numpy().astype(np.float32)

    return energy


def compute_pitch(wav, wav_sr):
    wav = wav.astype(np.float64)
    pitch, t = pyworld.dio(wav, wav_sr, frame_period=audio_params.hop_length / wav_sr * 1000)
    pitch = pyworld.stonemask(wav, pitch, t, audio_params.sampling_rate)
    return pitch


def process_wavs(train_config):
    text = process_text(train_config.data_path)
    
    wavs = sorted(os.listdir(train_config.wav_path))
    
    for i in tqdm(range(len(text))):
        duration = np.load(os.path.join(train_config.alignment_path, str(i)+".npy"))
        max_length = sum(duration)

        wav, sr = librosa.load(os.path.join(train_config.wav_path, wavs[i]))

        pitch = compute_pitch(wav, sr)
        pitch = pitch[:max_length]
        
        if np.sum(pitch != 0) <= 1:
            return None
        
        energy = compute_energy(wav)
        energy = energy[:max_length]

        os.makedirs(train_config.pitch_path, exist_ok=True)
        os.makedirs(train_config.energy_path, exist_ok=True)

        np.save(os.path.join(train_config.pitch_path, f"ljspeech-pitch-{i+1}.npy"), pitch)
        np.save(os.path.join(train_config.energy_path, f"ljspeech-energy-{i+1}.npy"), energy)


if __name__ == '__main__':
    process_wavs(train_config)