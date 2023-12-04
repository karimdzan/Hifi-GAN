import torch
import os
import wandb
import torchaudio
from src.model import Generator
from src.config import generator_config

def inference(generator_path, mel_spec, wav_path, max_wav_value, step, device, sample_rate):
    generator = Generator(generator_config).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device)['generator'])
    generator.eval()
    generator.remove_weight_norm()
    for filename in os.listdir(wav_path):
        wav = torchaudio.load(os.path.join(wav_path, filename))[0]
        mel = mel_spec(wav.to(device))
        wav_gen = generator(mel).squeeze(0).detach().cpu()
        # wav_gen = (wav_gen * max_wav_value).to(torch.int16)
        wandb.log(
            {
                filename.replace(".wav", ""): wandb.Audio(wav_gen, sample_rate),
            },
            step=step
        )
    generator.train()

