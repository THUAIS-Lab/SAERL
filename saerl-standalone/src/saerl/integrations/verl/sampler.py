"""Fixed dataset positions, distinct from stable sample IDs."""

import numpy as np
import torch
from verl.experimental.dataset.sampler import AbstractSampler


class SAERLSampler(AbstractSampler):
    def __init__(self, data_source, data_config):
        path = data_config.saerl.processed_file
        if not path:
            raise ValueError("data.saerl.processed_file is required for fixed-order sampling")
        values = torch.load(path, map_location="cpu", weights_only=True)
        positions = np.asarray(values)
        if positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer):
            raise TypeError("processed_file must contain a one-dimensional integer tensor of dataset positions")
        if len(positions) == 0 or np.any(positions < 0) or np.any(positions >= len(data_source)):
            raise ValueError("Sampler positions must be nonempty and within the filtered dataset")
        self.positions = positions.tolist()

    def __iter__(self):
        return iter(self.positions)

    def __len__(self):
        return len(self.positions)
