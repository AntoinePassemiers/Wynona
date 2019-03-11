# -*- coding: utf-8 -*-
# data_manager.py
# author : Antoine Passemiers

from deepcp.prot.parsers import parse_folder

import os
import copy
import random
import pickle
import json


class DataManager:

    def __init__(self, data_dir, temp_dir):
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        self.prot = dict()
        self.datasets = dict()

    def load_dataset(self, dataset='training_set'):
        with open(os.path.join(self.data_dir, 'sets.json'), 'r') as f:
            sets = json.load(f)
            self.datasets[dataset] = list()
            for prot_name in sets[dataset]:
                print(prot_name)
                self.datasets[dataset].append(prot_name)
                if not prot_name in self.prot.keys():
                    self.load_protein(prot_name)

    def load_protein(self, prot_name):
        if not os.path.exists(os.path.join(self.temp_dir, prot_name)):
            folder = self.find_protein_folder(prot_name)
            seqdata = parse_folder(os.path.join(self.data_dir, folder), prot_name)
            with open(os.path.join(self.temp_dir, prot_name), 'wb') as f:
                pickle.dump(seqdata, f)
        with open(os.path.join(self.temp_dir, prot_name), 'rb') as f:
            seqdata = pickle.load(f)
            self.prot[prot_name] = seqdata

    def find_protein_folder(self, prot_name):
        for subdir in os.listdir(self.data_dir):
            subdir = os.path.join(self.data_dir, subdir, prot_name)
            if os.path.isdir(subdir):
                return subdir
            print(subdir)
        assert(False)

    def proteins(self, dataset='train'):
        if not dataset in self.datasets.keys():
            self.load_dataset(dataset)
        for prot_name in self.datasets[dataset]:
            seqdata = self.prot[prot_name]
            yield seqdata
