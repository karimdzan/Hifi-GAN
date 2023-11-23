import torch
import numpy as np
import wandb
from src.trainer.synthesis import synthesis
from src.waveglow.waveglow.inference import inference
from src.waveglow.text import text_to_sequence


def prepare_texts(texts, text_cleaners, device='cuda:0'):
    result = []
    for raw_text in texts:
        text = np.array(text_to_sequence(raw_text, text_cleaners))
        text = np.stack([text])
        src_pos = np.array([i+1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().to(device)
        src_pos = torch.from_numpy(src_pos).long().to(device)
        result.append((sequence, src_pos, raw_text))
    return result


def get_data():
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]
    return tests


def create_logging_table(model, wave_glow, prepared_texts, step, device='cuda:0'):
    model.eval()
    for length_alpha in [0.8, 1.0, 1.3]:
        for pitch_alpha in [0.8, 1.0, 1.2]:
            for energy_alpha in [0.8, 1.0, 1.2]:
                for i, (sequence, src_pos, raw_text) in enumerate(prepared_texts):
                    mel = synthesis(model, sequence, src_pos, length_alpha, pitch_alpha, energy_alpha, device)

                    audio, sr = inference(
                        mel, wave_glow
                    )

                    wandb.log(
                      {
                        f"step-{step}-text-{i}-{length_alpha}_{pitch_alpha}_{energy_alpha}": wandb.Audio(
                            audio.astype("int16"), sr, 
                            caption=f"STEP - {step}; CONF - {length_alpha}_{pitch_alpha}_{energy_alpha}; TEXT - {raw_text}"
                        )
                      },
                    step=step)

    model.train()
