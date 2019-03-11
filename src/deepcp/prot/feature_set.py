# -*- coding: utf-8 -*-
# feature_set.py
# author : Antoine Passemiers

import numpy as np


class FeatureSet:

    def __init__(self, prot_name, msa, msa_weights, distances):
        self.prot_name = prot_name
        self.msa = msa
        self.msa_weights = msa_weights
        self.distances = distances
        self.features = {
            'global': { 'names': list(), 'values': list() },
            '1-dim': { 'names': list(), 'values': list() },
            '2-dim': { 'names': list(), 'values': list() }
        }
        self.ready_to_use = False

    def add(self, feature_name, data):
        assert(not self.ready_to_use)
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                self.features['global']['names'] += [feature_name] * len(data)
                self.features['global']['values'] += list(data)
            elif data.ndim == 2:
                self.features['1-dim']['names'] += [feature_name] * data.shape[0]
                self.features['1-dim']['values'].append(data)
            elif data.ndim == 3:
                self.features['2-dim']['names'] += [feature_name] * data.shape[0]
                self.features['2-dim']['values'].append(data)
            else:
                assert(0)
        elif isinstance(data, (int, float)):
            self.features['global']['names'].append(feature_name)
            self.features['global']['values'].append(data)
        else:
            assert(0)

    def concat(self):
        self.features['global']['values'] = np.asarray(self.features['global']['values'])
        self.features['1-dim']['values'] = np.concatenate(self.features['1-dim']['values'], axis=0)
        self.features['2-dim']['values'] = np.concatenate(self.features['2-dim']['values'], axis=0)
        self.features['2-dim']['values'][np.isnan(self.features['2-dim']['values'])] = 0 # TODO
        self.features['2-dim']['values'][np.isinf(self.features['2-dim']['values'])] = 0 # TODO
        self.ready_to_use = True
        return self
