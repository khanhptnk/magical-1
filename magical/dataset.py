import os
import json
import random
import logging
import pickle

import numpy as np


class Dataset:

    def __init__(self, config, splits=['train', 'val', 'test'], seed=None):

        self.random = random.Random(seed)

        self.data = {}
        self.data_dir = config.data_dir
        self.splits = splits
        for split in splits:
            self.data[split] = DataSplit(self._load_data(split), seed=seed)
            if split == 'train':
                if self.config.train_subset:
                    self.data[split] = self.data[split][:self.config.train_subset]
                self.data[split + '_val'] = DataSplit(self.random.sample(
                    self.data[split].data, max(1, len(self.data[split]) // 20)),
                    seed=seed)

        for split in splits:
            logging.info('Loaded %s set of size %d' % (split, len(self.data[split])))

    def _load_data(self, split):

        file_name = os.path.join(self.data_dir, split + '.json')

        with open(file_name, 'rb') as f:
            data = pickle.load(f)

        return data

    def __getitem__(self, split):
        return self.data[split]


class DataSplit:

    def __init__(self, data, seed=None):

        self.data = data
        self.random = random.Random(seed)
        self.random.shuffle(data)

    def __len__(self):
        return len(self.data)

    def random_item(self):
        return self.random.choice(self.data)

    def random_batch(self, batch_size):
        return self.random.sample(self.data, batch_size)

    def iterate_batches(self, batch_size=1, cycle=False):

        self.idx = 0

        while True:

            batch = self.data[self.idx : (self.idx + batch_size)]

            if len(batch) < batch_size:
                batch += self.random.sample(self.data, batch_size - len(batch))

            self.idx += batch_size
            if self.idx >= len(self.data):
                self.random.shuffle(self.data)
                self.idx = 0

            if batch:
                yield batch

            if not cycle and self.idx == 0:
                break

