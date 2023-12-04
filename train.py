from src.config import train_config, generator_config, discriminator_config, mel_config
from src.trainer import configure_training, train


generator, \
    mpd, msd, \
        generator_opt, \
            discriminator_opt, \
                generator_scheduler, \
                    discriminator_scheduler, \
                        dataloader, \
                          dataset, \
                             logger, \
                                 generator_scaler, \
                                 discriminator_scaler = configure_training(train_config, 
                                                        generator_config, 
                                                        discriminator_config, 
                                                        mel_config)
train(generator, 
      mpd, msd, 
      generator_opt, 
      discriminator_opt, 
      generator_scheduler, 
      discriminator_scheduler, 
      dataloader,
      dataset,
      logger,
      generator_scaler,
      discriminator_scaler)