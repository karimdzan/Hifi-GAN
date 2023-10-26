import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
        
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["duration"] = [item["duration"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]
    result_batch["audio"] = [item["audio"] for item in dataset_items]

    spec_max_size = max([item["spectrogram"].shape[2] for item in dataset_items])
    result_batch["spectrogram"] = torch.cat([
        F.pad(item["spectrogram"], (0, spec_max_size - item["spectrogram"].size(2)))
        for item in dataset_items
    ])

    result_batch["spectrogram_length"] = torch.tensor([
        item["spectrogram"].shape[2] for item in dataset_items
    ])

    te_max_size = max([item["text_encoded"].shape[1] for item in dataset_items])
    result_batch["text_encoded"] = torch.cat([
        F.pad(item["text_encoded"], (0, te_max_size - item["text_encoded"].size(1)))
        for item in dataset_items
    ])

    result_batch["text_encoded_length"] = torch.tensor([
        item["text_encoded"].shape[1] for item in dataset_items
    ])

    return result_batch