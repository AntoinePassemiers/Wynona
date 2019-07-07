# -*- coding: utf-8 -*-
# data_loader.py
# author : Antoine Passemiers

from torch.utils.data import DataLoader


class AdaptiveDataLoader(DataLoader):

    def __init__(self, **kwargs):
        kwargs['collate_fn'] = AdaptiveDataLoader.adaptive_collate_fn
        DataLoader.__init__(self, **kwargs)

    @staticmethod
    def adaptive_collate_fn(batch):
        batch = list(map(list, zip(*batch)))
        return batch
