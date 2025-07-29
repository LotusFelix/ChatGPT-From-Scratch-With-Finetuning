import torch
from torch.utils.data import Dataset

import tiktoken

from .utils import format_input

# Initialize tokenizer once
_tokenizer = tiktoken.get_encoding("gpt2")

class InstructionDataset(Dataset):
    """
    Dataset for instruction fine-tuning.
    Each example is tokenized and stored as a list of token IDs.
    """
    def __init__(self, data_list):
        self.data = data_list
        self.encoded = []
        for item in data_list:
            response = f"

### Response:
{item['output']}"
            full = format_input(item) + response
            self.encoded.append(_tokenizer.encode(full))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def custom_collate(batch, pad_token=50256, ignore_token=-100, max_length=None, device=None):
    """
    Collate function to pad and mask sequences for training.
    """
    max_len = max(len(seq) for seq in batch) + 1
    inputs, targets = [], []
    for seq in batch:
        seq = seq + [pad_token] * (max_len - len(seq))
        inp = torch.tensor(seq[:-1])
        tgt = torch.tensor(seq[1:])
        # Mask padding tokens in target
        mask = tgt == pad_token
        if mask.any():
            first_pad = mask.nonzero(as_tuple=True)[0][0]
            tgt[first_pad:] = ignore_token
        if max_length:
            inp, tgt = inp[:max_length], tgt[:max_length]
        inputs.append(inp)
        targets.append(tgt)
    inputs = torch.stack(inputs).to(device)
    targets = torch.stack(targets).to(device)
    return inputs, targets