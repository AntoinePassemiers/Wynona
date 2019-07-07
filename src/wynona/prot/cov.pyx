# -*- coding: utf-8 -*-
# cov.pyx
# author : Antoine Passemiers
# distutils: language=c
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: boundscheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from libc.stdlib cimport *


cdef int N_AA_TYPES = 22


def cov_from_multiple_alignment(cnp.uint8_t[:, :] sequences, float identity_threshold=0.38):
    """Compute covariance matrix for each residue pairing
    on the basis of a multiple sequence alignment.

    Parameters:
        sequences (np.ndarray):
            Array of shape (n_sequences, sequence_length) where
            n_sequences is the number of sequences in the multiple
            alignement and sequence_length is the length of the
            original amino acid sequence to be aligned.
        identity_threshold (float):
            Sequence identity threshold
    """
    cdef int a, b, i, j, k, remaining_residues
    cdef int n_sequences = sequences.shape[0]
    cdef int sequence_length = sequences.shape[1]
    cdef cnp.float_t[:, :] proba = np.ones(
        (sequence_length, N_AA_TYPES), dtype=np.float)
    cdef cnp.float_t[:, :, :, :] covmat = np.full(
        (sequence_length, sequence_length, N_AA_TYPES, N_AA_TYPES),
        1. / float(N_AA_TYPES), dtype=np.float)
    cdef cnp.int_t[:] counts = np.zeros(n_sequences, dtype=np.int)
    
    with nogil:
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                remaining_residues = <int>(sequence_length * identity_threshold)
                k = 0
                while remaining_residues >= 0 and k < sequence_length:
                    if sequences[i, k] != sequences[j, k]:
                        remaining_residues -= 1
                    k += 1

                    if remaining_residues > 0:
                        # Identity threshold exceeded
                        counts[i] += 1
                        counts[j] += 1
        
    cdef cnp.float_t[:] weights = 1.0 / (1.0 + np.asarray(counts, dtype=np.float))
    cdef cnp.float_t weight_sum = np.sum(weights)

    with nogil:
        for i in range(sequence_length):
            for k in range(n_sequences):
                proba[i, sequences[k, i]] += weights[k]
            for a in range(N_AA_TYPES):
                proba[i, a] /= <float>(N_AA_TYPES + weight_sum)
        for i in range(sequence_length):
            for j in range(sequence_length):
                for k in range(n_sequences):
                    covmat[i, j, sequences[k, i], sequences[k, j]] += weights[k]
                
        for i in range(sequence_length):
            for j in range(sequence_length):
                for a in range(N_AA_TYPES):
                    for b in range(N_AA_TYPES):
                        covmat[i, j, a, b] /= <float>(N_AA_TYPES + weight_sum)
                        covmat[i, j, a, b] -= (proba[i, a] * proba[j, b])
        
    return np.asarray(covmat)
