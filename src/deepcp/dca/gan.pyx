# -*- coding: utf-8 -*-
# simulated_annealing.pyx
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
from libc.string cimport memset


np_data_t = np.float32
ctypedef cnp.float32_t data_t
np_aa_t = np.uint8
ctypedef cnp.uint8_t aa_t


cdef struct HMatrix:
    data_t** data
    int n
    int q


cdef struct JTriangularMatrix:
    data_t** data
    int n
    int q


cdef HMatrix* create_H_mat(int n, int q):
    cdef HMatrix* mat = <HMatrix*>malloc(sizeof(HMatrix))
    mat.n = n
    mat.q = q
    mat.data = <data_t**>malloc(n * sizeof(data_t*))
    cdef data_t[:, :] data = np.random.normal(loc=0., scale=.05, size=(n, q)).astype(np_data_t)
    cdef int i, j
    with nogil:
        for i in range(n):
            mat.data[i] = <data_t*>malloc(q * sizeof(data_t))
            for j in range(q):
                mat.data[i][j] = data[i, j]
    return mat


cdef inline data_t get_H_mat_el(HMatrix* mat, int i, int k) nogil:
    return mat.data[i][k]


cdef inline void update_H_mat_el(HMatrix* mat, int i, int k, data_t el) nogil:
    mat.data[i][k] -= el


cdef JTriangularMatrix* create_J_tmat(int n, int q):
    cdef JTriangularMatrix* tmat = <JTriangularMatrix*>malloc(sizeof(JTriangularMatrix))
    tmat.n = n
    tmat.q = q
    tmat.data = <data_t**>malloc((((n+1)*n)//2) * sizeof(data_t*))
    cdef data_t[:, :] data = np.random.normal(loc=0., scale=.05, size=(((n+1)*n)//2, q**2)).astype(np_data_t)
    cdef int i, j
    with nogil:
        for i in range(((n+1)*n)//2):
            tmat.data[i] = <data_t*>malloc((q ** 2) * sizeof(data_t))
            for j in range(q**2):
                tmat.data[i][j] = data[i, j]
    return tmat


cdef inline data_t get_J_tmat_el(JTriangularMatrix* tmat, int i, int j, int k, int l) nogil:
    return tmat.data[(i*(i+1))//2+j][k*tmat.q+l]


cdef inline void update_J_tmat_el(JTriangularMatrix* tmat, int i, int j, int k, int l, data_t el) nogil:
    tmat.data[(i*(i+1))//2+j][k*tmat.q+l] -= el


cdef inline data_t sigmoid(data_t x) nogil:
    return 1. / (1. + libc.math.exp(-x))


cdef class Discriminator:

    cdef int L
    cdef HMatrix* H
    cdef JTriangularMatrix* J
    cdef float alpha
    cdef float beta

    def __cinit__(self, L, alpha=0.0002, beta=50.):
        self.L = L
        self.H = create_H_mat(L, 21)
        self.J = create_J_tmat(L, 21)
        self.alpha = alpha
        self.beta = beta
    
    def __dealloc__(self):
        pass # TODO
    
    def forward(self, sequences):
        cdef cnp.float32_t[:] out = np.empty(len(sequences), dtype=np.float32)
        cdef aa_t[:, :] _sequences = np.asarray(sequences, dtype=np_aa_t)
        cdef data_t S
        cdef int b, i, j, k, l
        with nogil:
            for b in range(_sequences.shape[0]):
                S = 0
                for i in range(self.L):
                    for j in range(i+1):
                        k, l = _sequences[b, i], _sequences[b, j]
                        S += get_J_tmat_el(self.J, i, j, k, l)
                for i in range(self.L):
                    k = _sequences[b, i]
                    S += get_H_mat_el(self.H, i, k)
                out[b] = sigmoid(S)
        return np.asarray(out)

    def backward(self, predictions, labels, sequences):
        cdef data_t[:] _predictions = np.asarray(predictions, dtype=np_data_t)
        cdef data_t[:] _labels = np.asarray(labels, dtype=np_data_t)
        cdef aa_t[:, :] _sequences = np.asarray(sequences, dtype=np_aa_t)
        cdef data_t[:] signal = np.empty(_sequences.shape[0], dtype=np_data_t)
        cdef data_t S, frob
        cdef int batch_size = _sequences.shape[0]
        cdef int b, i, j, k, l
        cdef int n = _sequences.shape[1]
        with nogil:
            for b in range(batch_size):
                signal[b] = _predictions[b] - _labels[b]
                signal[b] *= _predictions[b] * (1. - _predictions[b])

            # Update matrix H
            for b in range(batch_size):
                for i in range(self.L):
                    k = _sequences[b, i]
                    update_H_mat_el(self.H, i, k, self.alpha * signal[b] / batch_size)
            
            # Update matrix J
            for i in range(self.L):
                for j in range(i+1):
                    # Compute Frobenius norm
                    frob = 0
                    for k in range(21):
                        for l in range(21):
                            frob += get_J_tmat_el(self.J, i, j, k, l) ** 2
                    frob = libc.math.sqrt(frob)

                    for k in range(21):
                        for l in range(21):
                            S = 0
                            for b in range(batch_size):
                                if k == _sequences[b, i] and _sequences[b, j] == l:
                                    S += signal[b]
                            S /= batch_size
                            
                            # Regularization with Frobenius norm
                            S += self.beta * get_J_tmat_el(self.J, i, j, k, l) / frob

                            # Update parameter
                            update_J_tmat_el(self.J, i, j, k, l, self.alpha * S)
