import torchaudio.transforms as T
from torch import Tensor
import numpy as np

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self._aug = T.TimeMasking(*args, **kwargs)
        self.p = p

    def __call__(self, data: Tensor):
        if np.random.rand() < self.p:
            return self._aug(data)
        else:
            return data