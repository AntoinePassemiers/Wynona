# -*- coding: utf-8 -*-
# plmdca.pyx
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
from libc.stdio cimport printf
from libc.string cimport memset


np_data_t = np.float32
ctypedef cnp.float32_t data_t
np_aa_t = np.uint8
ctypedef cnp.uint8_t aa_t


cdef class PlmDCA:

    cdef float lambda_j, lambda_h, learning_rate
    cdef cnp.uint8_t[:, :] msa
    cdef data_t[:] weights
    cdef data_t B_eff
    cdef int B, N, q

    def __init__(self, _msa, _weights):
        self.q = 22
        self.lambda_j = 0.005
        self.lambda_h = 0.01
        self.msa = np.asarray(_msa, dtype=np.uint8)
        self.weights = np.asarray(_weights, dtype=np_data_t)
        self.B_eff = np.sum(self.weights)
        self.B = self.msa.shape[0]
        self.N = self.msa.shape[1]

    def init(self):
        J = np.random.normal(0, 1e-5, size=(self.N, self.N, self.q, self.q))
        H = np.random.normal(0, 1e-5, size=(self.N, self.q))
        return self.to_vector(J, H)

    def vector_to_parameters(self, x):
        mid = self.N * self.N * self.q * self.q
        J = x[:mid].reshape(self.N, self.N, self.q, self.q)
        H = x[mid:].reshape(self.N, self.q)
        return J, H

    def to_vector(self, J, H):
        return np.concatenate([np.asarray(J).flatten(), np.asarray(H).flatten()]).astype(np_data_t)

    def objective(self, x):
        _J, _H = self.vector_to_parameters(x)
        cdef data_t[:, :] H = _H.astype(np_data_t)
        cdef data_t[:, :, :, :] J = _J.astype(np_data_t)
        cdef cnp.float32_t temp, energy
        cdef cnp.float32_t[:] obj = np.zeros(self.N, dtype=np.float32)
        cdef int r, b, i, k, l
        with nogil:
            for r in range(self.N):
                for b in range(self.B):
                    temp = 0
                    for l in range(self.q):
                        energy = H[r, l]
                        for i in range(self.N):
                            if i != r:
                                energy += J[r, i, l, self.msa[b, i]]
                        temp += libc.math.exp(energy)
                    temp = -libc.math.log(temp)
                    temp += H[r, self.msa[b, r]]
                    for i in range(self.N):
                        if i != r:
                            temp += J[r, i, self.msa[b, r], self.msa[b, i]]
                    obj[r] -= self.weights[b] * temp
                obj[r] /= self.B_eff

        # L2 regularization
        l2_norm = self.lambda_h * np.sqrt(np.sum(np.asarray(H) ** 2))
        l2_norm += self.lambda_j * np.sqrt(np.sum(np.asarray(J) ** 2))
        return np.sum(np.asarray(obj) + l2_norm)

    cdef data_t cond_proba(self, data_t[:, :] H, data_t[:, :, :, :] J, cnp.uint8_t[:] sigma, int s, int r) nogil:
        cdef int i, l
        cdef data_t tmp
        
        # Compute numerator
        cdef data_t num = H[r, s]
        for i in range(self.N):
            if i != r:
                num += J[r, i, s, sigma[i]]
        num = libc.math.exp(num)
        # Compute denominator
        cdef data_t den = 0
        for l in range(self.q):
            tmp = H[r, l]
            for i in range(self.N):
                if i < r:
                    tmp += J[r, i, l, sigma[i]]
            den += libc.math.exp(tmp)
        return 0 if den == 0 else num / den

    def grad(self, x):
        _J, _H = self.vector_to_parameters(x)
        cdef data_t[:, :] H = _H.astype(np_data_t)
        cdef data_t[:, :, :, :] J = _J.astype(np_data_t)
        cdef data_t[:, :] grad_H = np.zeros((self.N, self.q), dtype=np_data_t)
        cdef data_t[:, :, :, :] grad_J = np.zeros((self.N, self.N, self.q, self.q), dtype=np_data_t)
        cdef int r, b, i, k
        cdef data_t[:, :] probas = np.empty((self.B, self.q), dtype=np_data_t)
        cdef data_t deriv
        with nogil:
            for r in range(self.N):

                printf('a')
                # Compute conditional probabilities
                for b in range(self.B):
                    for s in range(self.q):
                        probas[b, s] = self.cond_proba(H, J, self.msa[b], s, r)
                printf('b')

                # Compute partial derivatives of H_r(s) and perform step
                for s in range(self.q):
                    deriv = 0
                    for b in range(self.B):
                        deriv -= self.weights[b] * (<float>(self.msa[b, r] == s) - probas[b, s])
                    deriv /= self.B_eff
                    deriv += 2 * self.lambda_h * H[r, s]
                    grad_H[r, s] = deriv
                printf('c')

                # Compute partial derivatives of J_ri(s, k) and perform step
                for i in range(r):
                    for s in range(self.q):
                        for k in range(self.q):
                            deriv = 0
                            for b in range(self.B):
                                if self.msa[b, i] == k:
                                    deriv -= self.weights[b] * (<float>(self.msa[b, r] == s) - probas[b, s])
                            deriv /= self.B_eff
                            deriv += 2 * self.lambda_j * J[r, i, s, k]
                            grad_J[r, i, s, k] = grad_J[i, r, s, k] = deriv
                printf('d')
        return self.to_vector(grad_J, grad_H)
    
    def step(self):
        self.H = np.asarray(self.H) - self.learning_rate * np.asarray(self.grad_H)
        self.J = np.asarray(self.J) - self.learning_rate * np.asarray(self.grad_J)
    
    def get_couplings(self):
        return np.asarray(self.J)
    
    def get_fields(self):
        return np.asarray(self.H)
    
    def get_gradients(self):
        return np.asarray(self.grad_J), np.asarray(self.grad_H)
