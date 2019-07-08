# -*- coding: utf-8 -*-
# pssm.py
# author : Antoine Passemiers

from wynona.prot.sequence import *

import numpy as np


# web.expasy.org/docs/relnotes/relstat.html
# Probabilities of randomly finding each amino acid in the Uniprot database
DATABASE_AA_COMPOSITION = np.array(
    [0.0826, 0.0553, 0.0406, 0.0546, 0.0137, 0.0393,
     0.0674, 0.0708, 0.0227, 0.0593, 0.0965, 0.0582,
     0.0241, 0.0386, 0.0472, 0.0660, 0.0534, 0.0109,
     0.0292, 0.0687])
P = DATABASE_AA_COMPOSITION


def compute_pssm(sequences):
    n_sequences = len(sequences)
    num_symbols = len(DATABASE_AA_COMPOSITION)
    m = len(sequences[0])
    N = np.zeros((m, num_symbols), dtype=np.double)
    data = np.empty((n_sequences, m), dtype=AA_DTYPE)
    # 1) Counts of each amino acid at each position
    for p in range(n_sequences):
        data[p, :] = sequences[p].to_array()
    for u in range(m):                         # for each column
        for b in range(num_symbols):           # for each amino acid
            N[u, b] += (data[:, u] == b).sum()

    # 2) Frequencies of each amino acid at each position
    # First version (used for computing the PSSM matrix)
    F = N / float(n_sequences)
    # Second version, more stable (used for drawing WebLogos)
    F_stable = np.log(1.0 - (N / (n_sequences + 1.0))) / np.log(1.0 / (n_sequences + 1.0))

    # 3) Pseudo-counts of each amino acid at each position
    alpha = n_sequences # Used for WebLogos
    alphas = np.sum(data != AA_CHAR_TO_INT['-'], axis=0)
    beta = np.sqrt(n_sequences)
    # First version (based on F)
    Q = np.divide((np.multiply(alphas, F.T).T + beta * P).T, (alphas + beta)).T
    # Second version (based on F_stable)
    Q_stable = np.divide((np.multiply(alpha, F_stable.T).T + beta * P).T, (alpha + beta)).T

    # 4) PSSM Matrix : log-odds ratios of each amino acid at each position
    M = np.log2(np.divide(Q, P))
    return F, F_stable, Q, Q_stable, M
