from src.config import TrainConfig, FastSpeech2Config, MelSpectrogramConfig
from src.trainer import configure_training, train

model_config = FastSpeech2Config()
train_config = TrainConfig()
mel_config = MelSpectrogramConfig()

model, \
    train_loader, \
        fastspeech_loss, \
            optimizer, \
                scheduler, \
                    wave_glow, \
                         logger, \
                             scaler = configure_training(model_config, 
                                                   train_config, 
                                                   mel_config)
train(train_config, 
      model, 
      train_loader, 
      fastspeech_loss, 
      optimizer, 
      scheduler, 
      wave_glow,
      logger,
      scaler)