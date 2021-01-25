import torch
import numpy as np
import random

from torch.utils.data import sampler


class randomSequentialStratifiedSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.batch_size = batch_size
        self.num_samples = len(data_source)
        self.dataset = data_source
        self.indices_intervals = data_source.class_idxs_intervals
        assert (self.batch_size >= len(self.indices_intervals) and \
                (self.batch_size % len(self.indices_intervals)) == 0), \
            f"Batch size should be a multiple of number of classes ({data_source.n_classes})"

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        samples_on_each_class = self.batch_size // len(self.indices_intervals)
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)

        for i in range(n_batch):
            batch_samples = torch.LongTensor(self.batch_size).fill_(0)
            for idx, class_id in enumerate(self.indices_intervals):
                samples_start_idx, sample_end_idx = self.indices_intervals[class_id]

                batch_samples[idx * samples_on_each_class : (idx + 1) * samples_on_each_class] = \
                    torch.LongTensor(np.random.choice(torch.arange(samples_start_idx, sample_end_idx), samples_on_each_class))

            batch_samples = batch_samples[torch.randperm(self.batch_size)]

            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_samples
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.batch_size = batch_size
        self.num_samples = len(data_source)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)

        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size : (i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)


class batchOverfitSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size, ):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            batch_index = torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            tail_index = torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)