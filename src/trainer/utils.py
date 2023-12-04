import torch
import torch.nn.functional as F
import numpy as np
import random
import torch.backends.cudnn as cudnn
from src.config import mel_config

def init_torch_seeds(seed: int = 0):
    if seed == 0:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


