import argparse
import os
import torch
import torchaudio
from src.model import Generator
from src.config import GeneratorConfig
from scipy.io.wavfile import write
from src.config import train_config, mel_config
from src.data import MelWavDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        help="Path to generator checkpoint", 
        required=True)
    parser.add_argument(
        "--data", 
        type=str,
        help="Path to data", 
        required=True)
    parser.add_argument(
        "--type", 
        type=str,
        default='audio',
        help="Provide data type: audio or mels"
        )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results", 
        help="Provide an output folder")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    dataset = MelWavDataset(train_config, mel_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(GeneratorConfig()).to(device)
    print(generator)
    generator.load_state_dict(torch.load(args.checkpoint)["generator"])
    generator.eval()
    generator.remove_weight_norm()
    for data_file in os.listdir(args.data):
        if args.type == 'audio':
            wav = torchaudio.load(os.path.join(args.data, data_file))[0]
            mel = dataset.melspec_with_pad(wav.to(device))
            wav_gen = generator(mel).squeeze(0).detach().cpu().numpy()
            write(os.path.join(args.output, data_file), 22050, wav_gen)
        else:
            mel = torch.load(os.path.join(args.data, data_file)).unsqueeze(0).to(device)
            wav = generator(mel).squeeze(0).detach().cpu().numpy()
            write(os.path.join(args.output, data_file.replace(".pt", ".wav")), 22050, wav)

if __name__ == "__main__":
    main()
