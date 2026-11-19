import random
from typing import Iterator, List

import torch
from torch.utils.data import Sampler


def collate_fn(batch):
    if not batch:
        return batch
    return torch.utils.data.default_collate(batch)


class SingleDatasetBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        batch = []
        for index in indices:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class DistributedSingleDatasetBatchSampler(SingleDatasetBatchSampler):
    pass
