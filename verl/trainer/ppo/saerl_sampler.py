import torch
import numpy as np
from torch.utils.data import Sampler

class SAERLSampler(Sampler):
    def __init__(self, sort_list):
        self.sort_list = sort_list

    def __iter__(self):
        # 按给定顺序返回索引
        yield from self.sort_list

    def __len__(self):
        return len(self.sort_list)
