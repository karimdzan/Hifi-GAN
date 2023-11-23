import torch

def synthesis(model, text, src_pos, length_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0, device='cuda:0'):
    with torch.no_grad():
        mel = model.forward(text, src_pos, length_alpha=length_alpha, pitch_alpha=pitch_alpha, energy_alpha=energy_alpha)
    return mel.contiguous().transpose(1, 2).to(device)

