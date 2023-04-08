import os
import json
import random
import logging
import pickle

import numpy as np


class Dataset:

    def __init__(self, config, splits=['train', 'val', 'test'], prefix='', data_split_cls=None, seed=None):

        self.random = random.Random(seed)
        self.config = config
        self.data = {}
        self.data_dir = config.data_dir
        self.splits = splits
        self.prefix = prefix

        data_split_cls = data_split_cls or DataSplit
        for split in splits:
            self.data[split] = data_split_cls(self._load_data(split), seed=seed)
            if split == 'train':
                if self.config.train_subset:
                    self.data[split] = self.data[split][:self.config.train_subset]
                self.data[split + '_val'] = DataSplit(self.random.sample(
                    self.data[split].data, max(config.train.batch_size,
                                           min(100, len(self.data[split]) // 20))),
                    seed=seed)

        for split in self.data:
            logging.info('Loaded %s set of size %d' % (split, len(self.data[split])))

    def _load_data(self, split):

        file_name = os.path.join(self.data_dir, self.prefix + split + '.pkl')

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

    def shuffle(self):
        self.random.shuffle(self.data)

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


class VAEDataSplit(DataSplit):

    def __init__(self, data, seed=None):

        new_data = []
        for item in data:
            new_data.extend(item['observations'])
        super().__init__(new_data, seed=seed)
