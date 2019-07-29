# -*- coding: utf-8 -*-
# data_manager.py
# author : Antoine Passemiers

from wynona.parsers.parse import parse_folder

import os
import random
import json


class DataManager:

    def __init__(self, data_dir, all_in_memory=False):
        self.data_dir = data_dir
        assert(isinstance(all_in_memory, bool))
        self.all_in_memory = all_in_memory
        self.prot = dict()
        self.dataset = list()
        self.dataset_name = ''

    def load(self, dataset_name):
        with open(os.path.join(self.data_dir, 'sets.json'), 'r') as f:
            sets = json.load(f)
            self.dataset = list()
            for prot_name in sets[dataset_name]:
                self.dataset.append(prot_name)
        self.dataset_name = dataset_name
        return self

    def load_protein(self, prot_name):
        folder = self.find_protein_folder(prot_name)
        if prot_name in self.prot.keys():
            seqdata = self.prot[prot_name]
        else:
            seqdata = parse_folder(folder, prot_name)
        if self.all_in_memory:
            self.prot[prot_name] = seqdata
        return seqdata

    def find_protein_folder(self, prot_name):
        for subdir in os.listdir(self.data_dir):
            subdir = os.path.join(self.data_dir, subdir, prot_name)
            if os.path.isdir(subdir):
                return subdir
        assert(False)

    def all(self):
        batch = list()
        for i in range(len(self.dataset)):
            prot_name = self.dataset[i]
            batch.append(self.load_protein(prot_name))
        return batch

    def sample(self, batch_size):
        batch = list()
        for i in range(batch_size):
            prot_name = random.choice(self.dataset)
            batch.append(self.load_protein(prot_name))
        return batch
