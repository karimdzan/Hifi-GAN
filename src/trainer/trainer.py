import torch.nn.functional as F
import torch
from tqdm import tqdm
import os
from src.wandb_writer import WanDBWriter
import gc
from torch.cuda.amp import GradScaler
from torch import autocast
from src.config import train_config
from src.data import MelWavDataset
from src.model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from src.loss import discriminator_loss, generator_loss, feature_loss
from src.trainer.inference import inference
from src.trainer.utils import init_torch_seeds
import itertools


@torch.no_grad()
def get_grad_norm(model, norm_type=2):
    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
        ),
        norm_type,
    )
    return total_norm.item()

def collate(samples):
    return torch.cat([s[0] for s in samples], 0), torch.cat([s[1] for s in samples], 0)

def configure_training(train_config,  
                       generator_config, 
                       discriminator_config,
                       mel_config):
    
    init_torch_seeds(train_config.seed)
    os.makedirs(train_config.logs_path, exist_ok=True)

    dataset = MelWavDataset(train_config, mel_config)

    generator = Generator(generator_config).to(train_config.device)
    mpd = MultiPeriodDiscriminator(discriminator_config).to(train_config.device)
    msd = MultiScaleDiscriminator(discriminator_config).to(train_config.device)
    generator.train()
    mpd.train()
    msd.train()
    generator.count_params()
    mpd.count_params()
    msd.count_params()

    generator_opt = torch.optim.AdamW(generator.parameters(), lr=train_config.generator_lr, 
                                      betas=train_config.adam_betas)
    discriminator_opt = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), 
                                          train_config.discriminator_lr, betas=train_config.adam_betas)

    generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_opt, train_config.lr_decay)
    discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_opt, train_config.lr_decay)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate
    )

    logger = WanDBWriter(train_config)

    generator_scaler = GradScaler()
    discriminator_scaler = GradScaler()

    return generator, \
        mpd, msd, \
            generator_opt, \
                discriminator_opt, \
                    generator_scheduler, \
                        discriminator_scheduler, \
                            dataloader, \
                                dataset, \
                                    logger, \
                                        generator_scaler, \
                                        discriminator_scaler



def train(generator, 
          mpd, msd, 
          generator_opt, 
          discriminator_opt, 
          generator_scheduler, 
          discriminator_scheduler, 
          dataloader, 
          dataset, 
          logger, 
          generator_scaler, 
          discriminator_scaler
          ):
    
    init_torch_seeds(train_config.seed)

    pbar = tqdm(total=train_config.epochs * len(dataloader))

    steps = 0

    total_disc_loss = total_gen_mel_loss = total_gen_gan_loss = total_gen_fm_loss = total_gen_full_loss = 0.
    try:
        for epoch in range(train_config.epochs):
            for real_wav, real_mel in dataloader:
                steps += 1
                logger.set_step(steps)
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred_wav = generator(real_mel)
                    pred_mel = dataset.melspec_with_pad(pred_wav)
                    
                    discriminator_opt.zero_grad()
                    y_rs, y_gs, _, _ = mpd(real_wav, pred_wav.detach())
                    loss_disc_f = discriminator_loss(y_rs, y_gs)

                    y_rs, y_gs, _, _ = msd(real_wav, pred_wav.detach())
                    loss_disc_s = discriminator_loss(y_rs, y_gs)

                    loss_disc_all = loss_disc_s + loss_disc_f

                # loss_disc_all.backward()
                # discriminator_opt.step()
                discriminator_opt.zero_grad()
                discriminator_scaler.scale(loss_disc_all).backward()
                if steps % train_config.log_steps == 0:
                    logger.add_scalar("grad_norm_msd", get_grad_norm(msd))
                    logger.add_scalar("grad_norm_mpd", get_grad_norm(mpd))
                discriminator_scaler.step(discriminator_opt)
                discriminator_scaler.update()

                with autocast(device_type='cuda', dtype=torch.float16):
                    generator_opt.zero_grad()

                    loss_mel = F.l1_loss(real_mel, pred_mel) * 45

                    _, y_gs, fmap_rs, fmap_gs = mpd(real_wav, pred_wav)
                    _, y_hat_gs, fmap_s_rs, fmap_s_gs = msd(real_wav, pred_wav)

                    loss_fm_f = feature_loss(fmap_rs, fmap_gs) * 2
                    loss_fm_s = feature_loss(fmap_s_rs, fmap_s_gs) * 2

                    loss_gen_f = generator_loss(y_gs)
                    loss_gen_s = generator_loss(y_hat_gs)
                    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                # loss_gen_all.backward()
                # generator_opt.step()

                generator_opt.zero_grad()
                generator_scaler.scale(loss_gen_all).backward()
                if steps % train_config.log_steps == 0:
                    logger.add_scalar("grad_norm_gen", get_grad_norm(generator))
                generator_scaler.step(generator_opt)
                generator_scaler.update()

                total_disc_loss += loss_disc_all.item() / train_config.log_steps
                total_gen_mel_loss += loss_mel.item() / train_config.log_steps
                total_gen_gan_loss += (loss_disc_s + loss_gen_f).item() / train_config.log_steps
                total_gen_fm_loss += (loss_fm_f + loss_fm_s).item() / train_config.log_steps
                total_gen_full_loss += loss_gen_all.item() / train_config.log_steps

                torch.cuda.empty_cache()
                gc.collect()

                if steps % train_config.log_steps == 0:
                    logger.add_scalar("discriminator_loss", total_disc_loss)
                    logger.add_scalar("generator_mel_loss", total_gen_mel_loss)
                    logger.add_scalar("generator_gan_loss", total_gen_gan_loss)
                    logger.add_scalar("generator_feature_matching_loss", total_gen_fm_loss)
                    logger.add_scalar("generator_loss", total_gen_full_loss)
                    total_disc_loss = total_gen_mel_loss = total_gen_gan_loss = total_gen_fm_loss = total_gen_full_loss = 0.

                if steps % train_config.save_steps == 0:
                    save_path = os.path.join(train_config.logs_path, 'checkpoint_%d.pth' % steps)
                    torch.save({
                        'generator': generator.state_dict(), 
                        'mpd': mpd.state_dict(),
                        'msd': msd.state_dict(),
                        'step': steps
                        }, save_path
                    )

                    logger.add_image('True Mel', real_mel[0].detach().cpu().numpy().T)
                    logger.add_image('Pred Mel', pred_mel[0].detach().cpu().numpy().T)

                    inference(
                        save_path, dataset.melspec_with_pad, train_config.wav_path, train_config.max_wav_value,
                        steps, train_config.device, train_config.sample_rate
                    )
                
                pbar.update(1)

                discriminator_scheduler.step()
                generator_scheduler.step()

        pbar.finish()
    
    except KeyboardInterrupt:
        logger.wandb.finish()