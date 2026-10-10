"""Train the selected subset in its precomputed curriculum order."""

import torch
from verl.experimental.dataset.sampler import AbstractSampler


class SAERLSampler(AbstractSampler):
    """Yield full-dataset row indices exactly as stored in processed_file."""

    def __init__(self, data_source, data_config):
        self.sort_list = torch.load(
            data_config.saerl.processed_file, map_location="cpu", weights_only=True,
        ).tolist()

    def __iter__(self):
        yield from self.sort_list

    def __len__(self):
        return len(self.sort_list)
