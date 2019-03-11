# -*- coding: utf-8 -*-
# _profiles.pyx
# author : Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from libc.stdlib cimport *


cdef inline float _random() nogil:
    return <float>rand() / RAND_MAX


cdef inline int _wrandint(cnp.float32_t[:] weights) nogil:
    cdef float r = _random()
    cdef float cumsum = 0
    cdef int i
    for i in range(weights.shape[0]):
        cumsum += weights[i]
        if r <= cumsum:
            return i
    return i


def fit_pssm(cnp.uint8_t[:, :] msa, cnp.float32_t[:] weights, cnp.float32_t[:, :] freqs):
    cdef int i, j, k
    cdef int n_sequences = msa.shape[0]
    cdef int L = msa.shape[1]
    freqs[:, :] = 0
    cdef float d
    with nogil:
        for j in range(L):
            d = 0
            for i in range(n_sequences):
                freqs[j, msa[i, j]] += weights[i]
                d += weights[i]
            for k in range(freqs.shape[1]):
                freqs[j, k] /= d


def sample_pssm(cnp.float32_t[:, :] freqs, int n_samples):
    cdef int L = freqs.shape[0]
    cdef cnp.uint8_t[:, :] samples = np.empty((n_samples, L), dtype=np.uint8)
    cdef int i, j
    with nogil:
        for i in range(n_samples):
            for j in range(L):
                samples[i, j] = _wrandint(freqs[j])
    return np.asarray(samples)


def fit_hmm(cnp.uint8_t[:, :] msa, cnp.float32_t[:] weights, cnp.float32_t[:] pi, cnp.float32_t[:, :, :] a):
    cdef int i, j, k, l
    cdef int n_sequences = msa.shape[0]
    cdef int L = msa.shape[1]
    pi[:] = 0
    a[:, :, :] = 0
    cdef float d
    with nogil:
        d = 0
        for i in range(n_sequences):
            pi[msa[i, 0]] += weights[i]
            d += weights[i]
        for k in range(pi.shape[0]):
            pi[k] /= d
        
        for j in range(1, L):
            for i in range(n_sequences):
                a[j-1, msa[i, j-1], msa[i, j]] += weights[i]
            for k in range(pi.shape[0]):
                d = 0
                for l in range(pi.shape[0]):
                    d += a[j-1, k, l]
                for l in range(pi.shape[0]):
                    a[j-1, k, l] /= d



def sample_hmm(cnp.float32_t[:] pi, cnp.float32_t[:, :, :] a, int n_samples):
    cdef int L = a.shape[0] + 1
    cdef cnp.uint8_t[:, :] samples = np.empty((n_samples, L), dtype=np.uint8)
    cdef int i, j
    with nogil:
        for i in range(n_samples):
            samples[i, 0] = _wrandint(pi)
            for j in range(1, L):
                samples[i, j] = _wrandint(a[j-1, samples[i, j-1], :])
    return np.asarray(samples)
