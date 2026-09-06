import random
import torch
import numpy as np
from typing import Dict, Union


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Usage:
    def __init__(self, default=None):
        if default is None:
            self.storage = {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "n_completions": 0
            }
        else:
            self.storage = dict(default)

    def dict(self, *args, **kwargs) -> Dict:
        return self.storage

    def __repr__(self):
        return self.dict().__repr__()

    def __add__(self, other: Union['Usage', Dict]) -> 'Usage':
        if isinstance(other, Usage):
            other = other.dict()

        new_usage = Usage(self.dict())
        for key, value in other.items():
            if key in self.storage:
                new_usage[key] += value
            else:
                new_usage[key] = value

        return new_usage

    def __getitem__(self, item):
        return self.storage[item]

    def __setitem__(self, key, value):
        self.storage[key] = value
