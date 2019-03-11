# -*- coding: utf-8 -*-
# profiles.py
# author : Antoine Passemiers

from deepcp.prot.sequence import Sequence, N_STATES

import random
from abc import ABCMeta, abstractmethod
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from deepcp.prot._profiles import *


class ProfileSampler(metaclass=ABCMeta):

    def __init__(self, msa, weights):
        self.msa = np.asarray([seq.to_array(states=True) for seq in msa])
        self.weights = np.asarray(weights, dtype=np.float32)
        # self.weights = np.asarray(np.ones(len(weights)), dtype=np.float32)
        self.L = self.msa.shape[1]
    
    def sample(self, n_sequences, one_hot_encoded=True):
        data = self._sample(n_sequences)
        if one_hot_encoded:
            data = np.asarray([Sequence(data[i]).to_array(
                states=True, one_hot_encoded=True) for i in range(len(data))])
        return data
    
    @abstractmethod
    def _sample(self, n_sequences):
        pass


class UniformSampler(ProfileSampler):

    def __init__(self, *args, **kwargs):
        ProfileSampler.__init__(self, *args, **kwargs)

    def _sample(self, n_sequences):
        return np.random.randint(0, N_STATES, size=(n_sequences, self.L))


class PSSMSampler(ProfileSampler):

    def __init__(self, *args, **kwargs):
        ProfileSampler.__init__(self, *args, **kwargs)
        self.freqs = np.zeros((self.L, N_STATES), dtype=np.float32)
        fit_pssm(self.msa, self.weights, self.freqs)

    def _sample(self, n_sequences):
        return sample_pssm(self.freqs, n_sequences)


class HMMSampler(ProfileSampler):

    def __init__(self, *args, **kwargs):
        ProfileSampler.__init__(self, *args, **kwargs)
        self.pi = np.zeros(N_STATES, dtype=np.float32)
        self.a = np.zeros((self.L-1, N_STATES, N_STATES), dtype=np.float32)
        assert(not np.any(self.msa >= N_STATES))
        fit_hmm(self.msa, self.weights, self.pi, self.a)

    def _sample(self, n_sequences):
        return sample_hmm(self.pi, self.a, n_sequences)


class NeighbourhoodSampler(ProfileSampler):

    def __init__(self, *args, **kwargs):
        ProfileSampler.__init__(self, *args, **kwargs)
        self.weights /= self.weights.sum()
    
    def _sample(self, n_sequences):
        sequences = list()
        for i in range(n_sequences):
            seq = np.copy(self.msa[np.random.choice(np.arange(len(self.weights)), p=self.weights)])
            for k in range(7): # TODO
                site = np.random.randint(0, len(seq))
                aas = set(range(N_STATES)) - {seq[site]}
                seq[site] = random.choice(list(aas))
            sequences.append(seq)
        return np.asarray(sequences)


class MixtureSampler(ProfileSampler):

    def __init__(self, *args, **kwargs):
        ProfileSampler.__init__(self, *args, **kwargs)
        self.s1 = PSSMSampler(*args, **kwargs)
        self.s2 = HMMSampler(*args, **kwargs)
        self.s3 = UniformSampler(*args, **kwargs)
        self.s4 = NeighbourhoodSampler(*args, **kwargs)

    def _sample(self, n_sequences):
        sequences = list()
        for i in range(n_sequences):
            if random.random() <= 0.5:
                if random.random() < 0.4:
                    sequences.append(self.s3._sample(1)[0])
                else:
                    sequences.append(self.s1._sample(1)[0])
            else:
                sequences.append(self.s2._sample(1)[0])
        return np.asarray(sequences)
