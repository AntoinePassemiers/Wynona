# -*- coding: utf-8 -*-
# features.pyx
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


cdef cnp.float_t CNP_INF = <cnp.float_t>np.inf
cdef int N_AA_TYPES = 22
cdef cnp.double_t[:] UNIREF50_AA_FREQS = np.asarray([
    0.0,                   # (Gap)
    0.08419317947262096,   # A (Alanine)
    0.058285168924796175,  # R (Arginine)
    0.042046100985207224,  # N (Asparagine)
    0.05459461427571707,   # D (Aspartic acid)
    0.014738128673144382,  # C (Cysteine)
    0.06197693918988444,   # E/Z (Glutamine / glutamic acid)
    0.040134181479253114,  # Q (Glutamine)
    0.06674515639448204,   # G (Glycine)
    0.022640692546158938,  # H (Histidine)
    0.05453584555390309,   # I (Isoleucine)
    0.09559465355790236,   # L (Leucine)
    0.052128474295457077,  # K/O (Lysine / Pyrrolysine)
    0.021836718625738008,  # M (Methionine)
    0.038666461097043095,  # F (Phenylalanine)
    0.05077784852151325,   # P (Proline)
    0.07641242434788177,   # S (Serine)
    0.05738649269469764,   # T (Threonine)
    0.012712588510636218,  # W (Tryptophan)
    0.02941669643528118,   # Y (Tyrosine)
    0.06437953006998008,   # V (Valine)
    0.0007975607313478385, # X (Unknown)
    5.436173540314418e-07  # B (Asparagine / aspartic acid)
    ], dtype=np.double)
cdef cnp.double_t[:, :] ATCHLEY_FACTORS = np.asarray([
    [0., 0., 0., 0., 0.],
    [-0.591,-1.302,-0.733,1.570,-0.146], 
    [1.538,-0.055,1.502,0.440,2.897], 
    [0.945,0.828,1.299,-0.169,0.933], 
    [1.050,0.302,-3.656,-0.259,-3.242], 
    [-1.343,0.465,-0.862,-1.020,-0.255], 
    [1.357,-1.453,1.477,0.113,-0.837], 
    [0.931,-0.179,-3.005,-0.503,-1.853], 
    [-0.384,1.652,1.330,1.045,2.064], 
    [0.336,-0.417,-1.673,-1.474,-0.078], 
    [-1.239,-0.547,2.131,0.393,0.816], 
    [-1.019,-0.987,-1.505,1.266,-0.912], 
    [1.831,-0.561,0.533,-0.277,1.648], 
    [-0.663,-1.524,2.219,-1.005,1.212], 
    [-1.006,-0.590,1.891,-0.397,0.412], 
    [0.189,2.081,-1.628,0.421,-1.392], 
    [-0.228,1.399,-4.760,0.670,-2.647], 
    [-0.032,0.326,2.213,0.908,1.313], 
    [-0.595,0.009,0.672,-2.128,-0.184], 
    [0.260,0.830,3.097,-0.838,1.512], 
    [-1.337,-0.279,-0.544,1.242,-1.262],
    [0., 0., 0., 0., 0.] # TODO
    ], dtype=np.double)
cdef int N_AA_SYMBOLS = len(UNIREF50_AA_FREQS)


def compute_weights(msa, threshold):
    cdef float _threshold = threshold
    cdef cnp.uint8_t[:, :] _msa = np.asarray(msa, dtype=np.uint8)
    cdef int L = _msa.shape[1]
    cdef int n_sequences = _msa.shape[0]
    cdef int i, j, k
    cdef cnp.float_t[:] counts = np.ones(n_sequences, dtype=np.float)
    cdef int count
    with nogil:
        for i in range(n_sequences):
            for j in range(i):
                count = 0
                for k in range(L):
                    if _msa[i, k] == _msa[j, k]:
                        count += 1
                if <float>count / <float>L >= _threshold:
                    counts[i] += 1
                    counts[j] += 1
    return 1. / np.asarray(counts)


def count_amino_acids(msa, weights, pseudocount=1):
    cdef cnp.float_t _pseudocount = pseudocount
    cdef cnp.float_t[:] _weights = np.asarray(weights, dtype=np.float)
    cdef cnp.uint8_t[:, :] _msa = np.asarray(msa, dtype=np.uint8)
    cdef cnp.float_t _M_eff = np.sum(weights)
    cdef int i, j, k, l
    cdef int L = _msa.shape[1]
    cdef int n_sequences = _msa.shape[0]
    cdef cnp.float_t[:, :] p_i = np.zeros((L, N_AA_TYPES), dtype=np.float)
    cdef cnp.float_t[:, :, :, :] p_ij = np.zeros((L, L, N_AA_TYPES, N_AA_TYPES), dtype=np.float)
    with nogil:
        for i in range(L):
            for k in range(n_sequences):
                p_i[i, _msa[k, i]] += _weights[k]
        for i in range(L):
            for j in range(L):
                for k in range(n_sequences):
                    p_ij[i, j, _msa[k, i], _msa[k, j]] += _weights[k]
                for k in range(N_AA_TYPES):
                    for l in range(N_AA_TYPES):
                        p_ij[i, j, k, l] /= (n_sequences * n_sequences)
    F_i = ((_pseudocount / N_AA_TYPES) + np.asarray(p_i)) / (_M_eff + _pseudocount)
    F_ij = ((_pseudocount / (N_AA_TYPES**2)) + np.asarray(p_ij)) / (_M_eff + _pseudocount)
    return F_i, F_ij


cdef cnp.double_t compute_entropy(cnp.double_t[:] counts) nogil:
    cdef cnp.double_t total = 0
    cdef cnp.double_t entropy = 0
    cdef cnp.double_t proba
    cdef int i
    for i in range(counts.shape[0]):
        total += counts[i]
    for i in range(counts.shape[0]):
        if counts[i] > 0:
            proba = counts[i] / total
            entropy -= proba * libc.math.log2(proba)
    return entropy


cdef cnp.double_t[:, :] _couplings_to_cmap(J):
    cdef int i, j, k, l
    cdef cnp.float_t avg
    cdef cnp.float_t[:] avg_k = np.empty(J.shape[2], dtype=np.float)
    cdef cnp.float_t[:] avg_l = np.empty(J.shape[2], dtype=np.float)
    cdef cnp.float_t[:, :, :, :] _J = np.copy(J).astype(np.float)
    cdef int q = _J.shape[2]

    #J = J[:, :, 1:, 1:] # Ignore contributions from gaps
    with nogil:
        for i in range(_J.shape[0]):
            for j in range(i+1):
                avg = 0
                for l in range(q):
                    avg_k[l] = avg_l[l] = 0
                    for k in range(q):
                        avg_k[l] += _J[i, j, k, l]
                        avg_l[l] += _J[i, j, l, k]
                        avg += _J[i, j, k, l]
                    avg_k[l] /= q
                    avg_l[l] /= q
                avg /= (q ** 2)

                for k in range(q):
                    for l in range(q):
                        _J[i, j, k, l] += -avg_k[l] - avg_l[k] + avg    

    F = np.squeeze(np.linalg.norm(_J, axis=(2, 3)))
    #np.fill_diagonal(F, 0)
    F = (F + F.T) / 2.
    cdef cnp.double_t[:, :] f_ij = F.astype(np.double)
    F = _apply_apc(f_ij)
    return F


def couplings_to_cmap(J):
    return np.asarray(_couplings_to_cmap(J))


cdef cnp.double_t[:, :] _apply_apc(cnp.double_t[:, :] f_ij):
    cdef int n = f_ij.shape[0]
    cdef cnp.double_t[:, :] _f_ij_apc = np.empty_like(f_ij)
    cdef cnp.double_t[:] f_i = np.zeros(n, dtype=np.double)
    cdef cnp.double_t f = 0
    cdef int i, j
    with nogil:
        for i in range(n):
            for j in range(n):
                if i != j:
                    f_i[i] += f_ij[i, j]
                    f += f_ij[i, j]
            f_i[i] /= n
        f /= (n**2 - n)
    
    with nogil:
        if f != 0:
            for i in range(n):
                for j in range(i):
                    _f_ij_apc[i, j] = _f_ij_apc[j, i] = f_ij[i, j] - (f_i[i] * f_i[j]) / f
                _f_ij_apc[i, i] = 9999999 # TODO
    f_ij_apc = np.asarray(_f_ij_apc)
    np.fill_diagonal(f_ij_apc, np.min(f_ij_apc))
    return f_ij_apc


def apply_apc(f_ij):
    f_ij = np.copy(np.asarray(f_ij, dtype=np.double))
    #np.fill_diagonal(f_ij, 0)
    f_ij = (f_ij + f_ij.T) / 2.
    # f_ij = np.minimum(f_ij, f_ij.T)
    return np.asarray(_apply_apc(f_ij))


def extract_features_1d(cnp.uint8_t[:, :] msa, list feature_names):
    """Compute one-dimensional features.

    Parameters:
        msa (np.ndarray):
            Array of shape (n_sequences, sequence_length) where
            n_sequences is the number of sequences in the multiple
            alignement and sequence_length is the length of the
            original amino acid sequence to be aligned.
        feature_names (list):
            Names of the features to be extracted.
    """

    cdef cnp.double_t[:, :] counts
    cdef cnp.double_t[:, :] self_info, partial_entropies
    cdef cnp.double_t[:] count_x, count_y, entropies
    cdef cnp.double_t total_count, p_xy, p_x, p_y, prod
    cdef int i, j, k, l, a, b

    features = list()
    if len(set.intersection({'self-information', 'partial-entropy'}, set(feature_names))) > 0:

        # Compute amino acid frequencies
        counts = np.zeros((msa.shape[1], N_AA_SYMBOLS), dtype=np.double)
        with nogil:
            for i in range(msa.shape[0]):
                for j in range(msa.shape[1]):
                    counts[j, msa[i, j]] += 1
        UNIREF50_AA_FREQS[0] = (1. + np.sum(counts[:, 0])) / <cnp.double_t>(msa.shape[0] * msa.shape[1])
        with nogil:
            for j in range(msa.shape[1]):
                for k in range(N_AA_SYMBOLS):
                    counts[j, k] += 1 # Add pseudo-count
                    counts[j, k] /= (msa.shape[0] + N_AA_SYMBOLS)
        
        # Compute self-information
        if 'self-information' in feature_names:
            self_info = np.empty((msa.shape[1], 2*N_AA_SYMBOLS), dtype=np.double)
            with nogil:
                for i in range(msa.shape[1]):
                    for k in range(N_AA_SYMBOLS):
                        self_info[i, k] = libc.math.log2(counts[i, k] / UNIREF50_AA_FREQS[k])
                    for l in range(N_AA_SYMBOLS): # TO REMOVE
                        self_info[i, N_AA_SYMBOLS+l] = libc.math.log2(counts[i, l] / UNIREF50_AA_FREQS[l])
            features.append(np.asarray(self_info, dtype=np.float32))
        
        # Compute partial entropies
        if 'partial-entropy' in feature_names:
            partial_entropies = np.empty((msa.shape[1], 2*N_AA_SYMBOLS), dtype=np.double)
            with nogil:
                for i in range(msa.shape[1]):
                    for j in range(i+1):
                        for k in range(N_AA_SYMBOLS):
                            partial_entropies[i, k] = counts[i, l] * libc.math.log2(counts[i, k] / UNIREF50_AA_FREQS[k])
                        for l in range(N_AA_SYMBOLS):
                            partial_entropies[i, N_AA_SYMBOLS+l] = counts[j, l] * libc.math.log2(counts[j, l] / UNIREF50_AA_FREQS[l])
            features.append(np.nan_to_num(np.asarray(partial_entropies, dtype=np.float32)))        
    return features


def extract_features_2d(cnp.uint8_t[:, :] msa, list feature_names):
    """Compute two-dimensional features.

    Parameters:
        msa (np.ndarray):
            Array of shape (n_sequences, sequence_length) where
            n_sequences is the number of sequences in the multiple
            alignement and sequence_length is the length of the
            original amino acid sequence to be aligned.
        feature_names (list):
            Names of the features to be extracted.
    """

    cdef cnp.double_t[:, :] counts
    cdef cnp.double_t[:, :] mutual_information, normalized_mutual_information, cross_entropies
    cdef cnp.double_t[:] count_x, count_y, entropies
    cdef cnp.double_t total_count, p_xy, p_x, p_y, prod
    cdef int i, j, k, l, a, b

        
    # Compute mutual information
    features = list()
    counts = np.zeros((N_AA_SYMBOLS, N_AA_SYMBOLS), dtype=np.double)
    count_x = np.zeros(N_AA_SYMBOLS, dtype=np.double)
    count_y = np.zeros(N_AA_SYMBOLS, dtype=np.double)
    mutual_information = np.zeros((msa.shape[1], msa.shape[1]), dtype=np.double)
    with nogil:
        for i in range(msa.shape[1]):
            for j in range(i+1):
                count_x[:] = 0
                count_y[:] = 0
                counts[:, :] = 0
                for k in range(msa.shape[0]):
                    count_x[msa[k, i]] += 1
                    count_y[msa[k, j]] += 1
                    counts[msa[k, i], msa[k, j]] += 1
                total_count = 0
                for a in range(count_x.shape[0]):
                    total_count += count_x[a]
                mutual_information[i, j] = 0
                for a in range(count_x.shape[0]):
                    for b in range(count_x.shape[0]):
                        p_xy = counts[a, b] / total_count
                        p_x = count_x[a] / total_count if total_count > 0 else 0
                        p_y = count_y[b] / total_count if total_count > 0 else 0
                        if p_xy > 0 and p_x * p_y > 0:
                            mutual_information[i, j] += p_xy * libc.math.log2(p_xy / (p_x * p_y))
        for i in range(msa.shape[1]):
            for j in range(i+1):
                mutual_information[j, i] = mutual_information[i, j]
    if 'mutual-information' in feature_names:
        features.append(np.asarray(
            _apply_apc(mutual_information), dtype=np.float32)[np.newaxis, ...])
    
    if len(set.intersection({'normalized-mutual-information', 'cross-entropy'}, set(feature_names))) > 0:
        entropies = np.zeros(msa.shape[1], dtype=np.double)
        for i in range(msa.shape[1]):
            count_x[:] = 0
            for k in range(msa.shape[0]):
                count_x[msa[k, i]] += 1
            entropies[i] = compute_entropy(count_x)
    
    # Compute normalized mutual information
    if 'normalized-mutual-information' in feature_names:
        normalized_mutual_information = np.zeros((msa.shape[1], msa.shape[1]), dtype=np.double)
        for i in range(msa.shape[1]):
            for j in range(msa.shape[1]):
                prod = entropies[i] * entropies[j]
                if prod > 0:
                    normalized_mutual_information[i, j] = mutual_information[i, j] / prod
                else:
                    normalized_mutual_information[i, j] = 0
        features.append(np.asarray(
            _apply_apc(normalized_mutual_information), dtype=np.float32)[np.newaxis, ...])
    
    # Compute cross-entropies
    if 'cross-entropy' in feature_names:
        cross_entropies = np.zeros((msa.shape[1], msa.shape[1]), dtype=np.double)
        for i in range(msa.shape[1]):
            for j in range(msa.shape[1]):
                cross_entropies[i, j] = entropies[i] + entropies[j] - mutual_information[i, j]
        features.append(np.asarray(cross_entropies, dtype=np.float32)[np.newaxis, ...])

    return features
