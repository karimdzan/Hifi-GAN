import argparse
import os
import torch
import json
from scipy.io.wavfile import write, read
from tqdm import tqdm
from src.model.fastspeech import FastSpeech
from src.config import MelSpectrogramConfig, FastSpeech2Config, TrainConfig
from src.trainer.synthesis import synthesis
from src.trainer.utils import prepare_texts
from src.waveglow.waveglow.inference import inference
import src.waveglow.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", 
                        type=str, 
                        required=True, 
                        help="model checkpoint path")
    parser.add_argument("--text", 
                        type=str, 
                        required=True, 
                        help="path to your txt file")
    parser.add_argument("--duration", 
                        help="duration value, for example 1 or 0.8", 
                        default=1.)
    parser.add_argument("--pitch", 
                        help="pitch value, for example 1 or 0.8 ", 
                        default=1.)
    parser.add_argument("--energy", 
                        help="energy value, for example 1 or 0.8", 
                        default=1.)
    parser.add_argument("--save-path", 
                        default="./results/", 
                        help="path to save the speech")
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)

    model = FastSpeech(FastSpeech2Config, MelSpectrogramConfig)
    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model = model.to(args.device)

    wave_glow = utils.get_WaveGlow(args.device).to(args.device)

    texts = []
    if args.text is not None:
        with open(args.text) as f:
            for test in f.readlines():
                texts.append(test)

    prepared_texts = prepare_texts(texts, TrainConfig.text_cleaners, args.device)

    os.makedirs(args.save_path, exist_ok=True)

    model.eval()

    for i, (text, pos, _) in enumerate(prepared_texts):
        name = f"text-{i}-{args.duration}-{args.pitch}-{args.energy}.wav"
        mel = synthesis(model, text, pos, args.duration, args.pitch, args.energy)
        audio, sr = inference(mel, wave_glow)
        write(os.path.join(args.save_path, f"{name}.wav"), sr, audio.astype("int16"))                


if __name__ == "__main__":
    main()