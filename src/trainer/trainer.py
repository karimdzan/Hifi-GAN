from torch.utils.data import DataLoader
from torch.optim.lr_scheduler  import OneCycleLR
from torch import nn
import torch
from tqdm import tqdm
import os
from src.data import get_data_to_buffer, BufferDataset, collate_fn_tensor
from src.wandb_writer import WanDBWriter
from src.model.fastspeech import FastSpeech
from src.waveglow import utils as waveglow_utils
from src.trainer import utils as trainer_utils
from src.loss import FastSpeechLoss
from functools import partial
import gc
from torch.cuda.amp import GradScaler
from torch import autocast


def configure_training(model_config, train_config, mel_config):
    model = FastSpeech(model_config, mel_config)
    model = model.to(train_config.device)

    os.makedirs(train_config.checkpoint_path, exist_ok=True)

    fastspeech_loss = FastSpeechLoss(model_config=model_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scaler = GradScaler()

    buffer = get_data_to_buffer(train_config)
    dataset = BufferDataset(buffer)
    train_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_tensor, batch_expand_size=train_config.batch_expand_size),
        drop_last=True,
        num_workers=0
    )

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(train_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })

    wave_glow = waveglow_utils.get_WaveGlow()

    for path in os.listdir("."):
        if path.endswith(".patch"):
            os.remove(path)

    wave_glow = wave_glow.to(train_config.device)
    
    logger = WanDBWriter(train_config)

    return model, train_loader, fastspeech_loss, optimizer, scheduler, wave_glow, logger, scaler


def train(train_config, 
          model, 
          train_loader, 
          fastspeech_loss, 
          optimizer, 
          scheduler,
          wave_glow,
          logger, 
          scaler
          ):

    current_step = 0

    tqdm_bar = tqdm(total=train_config.epochs * len(train_loader) * train_config.batch_expand_size)

    prepared_texts = trainer_utils.prepare_texts(trainer_utils.get_data(), 
                                                 train_config.text_cleaners, 
                                                 device=train_config.device)

    t_l = m_l = d_l = p_l = e_l = 0.

    try:
        for epoch in range(train_config.epochs):
            for i, batchs in enumerate(train_loader):
                # real batch start here
                for j, db in enumerate(batchs):
                    current_step += 1
                    tqdm_bar.update(1)
                    logger.set_step(current_step)

                    # Get Data
                    character = db["text"].long().to(train_config.device)
                    mel_target = db["mel_target"].float().to(train_config.device)
                    duration = db["duration"].int().to(train_config.device)
                    pitch = db["pitch"].float().to(train_config.device)
                    energy = db["energy"].float().to(train_config.device)
                    mel_pos = db["mel_pos"].long().to(train_config.device)
                    src_pos = db["src_pos"].long().to(train_config.device)
                    max_mel_len = db["mel_max_len"]

                    # Forward
                    with autocast(device_type='cuda', dtype=torch.float16):
                        mel_output, \
                        duration_predictor_output, \
                        pitch_predictor_output, \
                        energy_predictor_output = model(character,
                                                        src_pos,
                                                        mel_pos=mel_pos,
                                                        mel_max_length=max_mel_len,
                                                        length_target=duration,
                                                        pitch_target=pitch,
                                                        energy_target=energy)

                        # Calc Loss
                        mel_loss, duration_loss, pitch_loss, energy_loss = \
                            fastspeech_loss(mel_output,
                                            duration_predictor_output,
                                            pitch_predictor_output,
                                            energy_predictor_output,
                                            mel_target,
                                            duration,
                                            pitch,
                                            energy)

                        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                    # Logger
                    t_l += total_loss.detach().cpu().numpy()
                    m_l += mel_loss.detach().cpu().numpy()
                    d_l += duration_loss.detach().cpu().numpy()
                    p_l += pitch_loss.detach().cpu().numpy()
                    e_l += energy_loss.detach().cpu().numpy()

                    if current_step % train_config.log_step == 0:
                        logger.add_scalar("duration_loss", d_l / train_config.log_step)
                        logger.add_scalar("mel_loss", m_l / train_config.log_step)
                        logger.add_scalar("pitch_loss", p_l / train_config.log_step)
                        logger.add_scalar("energy_loss", e_l / train_config.log_step)
                        logger.add_scalar("total_loss", t_l / train_config.log_step)
                        t_l = m_l = d_l = p_l = e_l = 0.
                        
                    # Backward
                    scaler.scale(total_loss).backward()

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.grad_clip_thresh)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    
                    torch.cuda.empty_cache()
                    gc.collect()

                    if current_step % train_config.save_step == 0:
                        os.makedirs(train_config.checkpoint_path, exist_ok=True)
                        torch.save({'model': model.state_dict(), 
                                    'optimizer': optimizer.state_dict()}, 
                                    os.path.join(train_config.checkpoint_path, 
                                                 'checkpoint_%d.pth.tar' % current_step))
                        table = trainer_utils.create_logging_table(model, wave_glow, prepared_texts, current_step, train_config.device)
                        logger.wandb.log({"examples": table})
                        
                        print("save model at step %d ..." % current_step)

    except KeyboardInterrupt:
        logger.wandb.finish()

    return model