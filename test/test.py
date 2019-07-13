# -*- coding: utf-8 -*-
# test.py
# author : Antoine Passemiers

from wynona.prot.align import align, align_to_itself
from wynona.prot.sequence import Sequence
from wynona.prot.nw import NeedlemanWunsch


def test_alignment():
    seq_a = Sequence('VEVLLGGDDGSLAFLPG')
    seq_b = Sequence('VVGDDGHYAPG')
    aligned_seq_a, aligned_seq_b = NeedlemanWunsch().align(seq_a, seq_b)
    assert(aligned_seq_a == 'VEVLLGGDDGSLAFLPG')
    assert(aligned_seq_b == 'V-V---GDDGHYA--PG')


def test_trim_prealignment():
    fixed_length_seq = 'VEVLLGGDDGSLAFLPG'
    aligned_seq = '----------VEVLMGGD--DGSLKFLMHPG----'
    aligned_seq = align(fixed_length_seq, aligned_seq)
    assert(aligned_seq == 'VEVLMGGDDGSLKFLPG')
