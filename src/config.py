from dataclasses import dataclass
import torch


@dataclass
class MelSpectrogramConfig:
    num_mels = 80


@dataclass
class FastSpeech2Config:
    vocab_size = 300
    max_seq_len = 3000

    pitch_vocab = 256
    energy_vocab = 256
    
    pitch_max = 862
    pitch_min = 0
    energy_max = 314
    energy_min = 0
    max_duration = 74

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    fft_conv1d_kernel = [9, 1]
    fft_conv1d_padding = [4, 0]

    filter_length = 1024
    win_length = 1024
    
    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1
    
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = ''
    UNK_WORD = ''
    BOS_WORD = ''
    EOS_WORD = ''


@dataclass
class TrainConfig:
    checkpoint_path = "./data/fastspeech2_checkpoints"
    logger_path = "./data/logger"
    mel_ground_truth = "./data/mels"
    alignment_path = "./data/alignments"
    data_path = './data/train.txt'
    pitch_path = './data/pitch'
    energy_path = './data/energy'
    wav_path = './data/LJSpeech-1.1/wavs'
    
    wandb_project = 'hw_tts'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 5000
    log_step = 10
    clear_Time = 20

    batch_expand_size = 32


mel_config = MelSpectrogramConfig()
model_config = FastSpeech2Config()
train_config = TrainConfig()