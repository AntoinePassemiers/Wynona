# -*- coding: utf-8 -*-
# check_homology.py
# author : Antoine Passemiers

from parse_fasta import parse_fasta

import os
import json
import numpy as np

import minineedle # https://github.com/scastlara/minineedle
import miniseq # https://github.com/scastlara/miniseq



if __name__ == '__main__':

    with open('sets.json', 'r') as f:
        sets = json.load(f)

    sequences = dict()

    for dp, dn, filenames in os.walk('.'):
        for f in filenames:
            if f == 'sequence.fa':
                data = parse_fasta(os.path.join(dp, f))
                name = os.path.basename(dp)
                sequences[name] = data['sequences'][0]

    set_a = sets['training_set']
    set_b = sets['benchmark_set_membrane']
    homologies = np.empty((len(set_a), len(set_b)), dtype=np.float)
    for idx_a, a in enumerate(set_a):
        for idx_b, b in enumerate(set_b):
            seq_a = miniseq.Protein(a, sequences[a])
            seq_b = miniseq.Protein(b, sequences[b])
            alignment = minineedle.Needleman(seq_a, seq_b)
            alignment.align()
            seq_a, seq_b = alignment.alseq1, alignment.alseq2
            n = float(len(seq_a))
            identity = sum(i == j for i, j in zip(seq_a, seq_b)) / n
            homologies[idx_a, idx_b] = identity
    print(np.max(homologies))
    print(np.min(homologies))
    print(np.mean(homologies))
