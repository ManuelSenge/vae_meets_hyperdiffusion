from torch.utils.data import Sampler
import torch
from typing import Iterator, List

class SingleInstanceBatchSampler(Sampler[List[int]]):
    def __init__(self, total_length) -> None:
        self.total_length = total_length

    def __len__(self) -> int:
        return self.total_length
    
    def __iter__(self) -> Iterator[int]:
        indices = iter([0] * self.total_length)
        yield from indices


if __name__ == '__main__':
    batch_sampler = SingleInstanceBatchSampler(30)

    for i, batch in enumerate(batch_sampler):
        print(i, batch)