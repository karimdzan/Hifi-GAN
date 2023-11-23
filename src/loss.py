import torch
import torch.nn as nn
    

class FastSpeechLoss(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.max_log_pitch = torch.log1p(torch.tensor(model_config.max_pitch))
        self.max_energy = model_config.max_energy
        self.max_log_duration = torch.log1p(torch.tensor(model_config.max_duration))

    def forward(self, mel, 
                      duration_predicted, 
                      pitch_predicted, 
                      energy_predicted, 
                      mel_target, 
                      duration_predictor_target, 
                      pitch_predictor_target, 
                      energy_predictor_target):

        mel_loss = self.l1_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                               torch.log1p(duration_predictor_target.float()))
        
        pitch_loss = self.mse_loss(pitch_predicted, 
                                   torch.log1p(pitch_predictor_target).float())
        
        energy_loss = self.mse_loss(energy_predicted,
                                    torch.log1p(energy_predictor_target).float())
        
        return mel_loss, duration_predictor_loss, pitch_loss, energy_loss