# -*- coding: utf-8 -*-
# nw.pyx
# author : Antoine Passemiers
# distutils: language=c
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: boundscheck=False

from wynona.prot.sequence import Sequence, AA_CHAR_TO_INT

import numpy as np
cimport numpy as cnp
cnp.import_array()

cimport libc.math
from libc.stdlib cimport *

ctypedef cnp.float32_t score_t
ctypedef cnp.int8_t seq_t
np_score_t = np.float32
np_seq_t = np.int8

cdef int GAP = AA_CHAR_TO_INT['-']

cdef int NO_POINTER = 2
cdef int POINTER_UP = -1
cdef int POINTER_LEFT = 1
cdef int POINTER_DIAG = 0


cdef class NeedlemanWunsch:
    """Efficient implementation of Needleman-Wunsch that
    allows only match, mismatch and gap penalties.
    Use this implementation on pre-aligned sequences only.
    """

    cdef score_t _match_score
    cdef score_t _mismatch_score
    cdef score_t _gap_score
    cdef score_t[:, :] _a_matrix
    cdef cnp.int8_t[:, :] _p_matrix

    def __cinit__(self, match=1, mismatch=-1, gap=-1):
        self._match_score = match
        self._mismatch_score = mismatch
        self._gap_score = gap

    def align(self, seq1, seq2):
        seq1 = seq1.to_array().astype(np_seq_t)
        seq2 = seq2.to_array().astype(np_seq_t)
        n, m = len(seq2), len(seq1)
        self._init_pointers_matrix(n, m)
        self._init_alignment_matrix(n, m)
        self._compute_cell_scores(seq1, seq2)
        aligned_seq1, aligned_seq2 = self._trace_back_alignment(seq1, seq2)
        return Sequence(aligned_seq1), Sequence(aligned_seq2)

    cdef void _init_pointers_matrix(self, int n, int m):
        cdef int i, j
        self._p_matrix = np.full((n + 1, m + 1), NO_POINTER, dtype=np.int8)
        self._p_matrix[1:, 0] = POINTER_UP
        self._p_matrix[0, 1:] = POINTER_LEFT

    cdef void _init_alignment_matrix(self, int n, int m):
        cdef int i, j
        self._a_matrix = np.zeros((n + 1, m + 1), dtype=np_score_t)
        with nogil:
            for i in range(1, n + 1):
                self._a_matrix[0, i] = self._a_matrix[0, i - 1] + self._gap_score
            for j in range(1, m + 1):
                self._a_matrix[j, 0] = self._a_matrix[j - 1, 0] + self._gap_score

    cdef void _compute_cell_scores(self, seq_t[:] seq1, seq_t[:] seq2):
        cdef int i, j
        cdef int n = seq2.shape[0]
        cdef int m = seq1.shape[0]
        cdef score_t top_score, diag_score, left_score
        with nogil:
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    top_score = self._a_matrix[i - 1, j] + self._gap_score
                    left_score = self._a_matrix[i, j - 1] + self._gap_score
                    diag_score = self._a_matrix[i - 1, j - 1]
                    diag_score += (self._match_score if (seq1[j - 1] == seq2[i - 1]) else self._mismatch_score)
                    if diag_score >= top_score:
                        if diag_score >= left_score:
                            self._p_matrix[i, j] = POINTER_DIAG
                            self._a_matrix[i, j] = diag_score
                        else:
                            self._p_matrix[i, j] = POINTER_LEFT
                            self._a_matrix[i, j] = left_score
                    elif top_score > left_score:
                        self._p_matrix[i, j] = POINTER_UP
                        self._a_matrix[i, j] = top_score
                    else:
                        self._p_matrix[i, j] = POINTER_LEFT
                        self._a_matrix[i, j] = left_score

    def _trace_back_alignment(self, seq_t[:] seq1, seq_t[:] seq2):

        cdef int n = seq2.shape[0]
        cdef int m = seq1.shape[0]
        cdef int i = n
        cdef int j = m
        cdef int cur = 0
        cdef seq_t[:] aligned_seq1 = np.empty(n + m, dtype=np_seq_t)
        cdef seq_t[:] aligned_seq2 = np.empty(n + m, dtype=np_seq_t)

        with nogil:
            while True:
                if self._p_matrix[i, j] == POINTER_DIAG:
                    aligned_seq1[cur] = seq1[j - 1]
                    aligned_seq2[cur] = seq2[i - 1]
                    i -= 1
                    j -= 1
                elif self._p_matrix[i, j] == POINTER_UP:
                    aligned_seq1[cur] = GAP
                    aligned_seq2[cur] = seq2[i - 1]
                    i -= 1
                elif self._p_matrix[i, j] == POINTER_LEFT:
                    aligned_seq1[cur] = seq1[j - 1]
                    aligned_seq2[cur] = GAP
                    j -= 1
                else:
                    break
                cur += 1
        cdef seq_t[:] aligned_seq1_trimmed = np.empty(cur, dtype=np_seq_t)
        cdef seq_t[:] aligned_seq2_trimmed = np.empty(cur, dtype=np_seq_t)
        with nogil:
            for i in range(cur):
                aligned_seq1_trimmed[i] = aligned_seq1[cur - 1 - i]
                aligned_seq2_trimmed[i] = aligned_seq2[cur - 1 - i]
        return np.asarray(aligned_seq1_trimmed), np.asarray(aligned_seq2_trimmed)
