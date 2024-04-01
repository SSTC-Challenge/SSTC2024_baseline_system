import torch, random
from torch.utils.data import RandomSampler, SequentialSampler

class WavBatchSampler(object):
    def __init__(self, dataset, tlen_range, shuffle=False, batch_size=1, drop_last=False):
        self.tlen_range = tlen_range
        self.batch_size = batch_size
        self.drop_last = drop_last

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

    def _renew(self):
        return [], random.uniform(self.tlen_range[0], self.tlen_range[1])

    def __iter__(self):
        batch, tlen = self._renew()
        for idx in self.sampler:
            batch.append((idx, tlen))
            if len(batch) == self.batch_size:
                yield batch
                batch, tlen = self._renew()
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size