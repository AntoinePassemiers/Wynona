# -*- coding: utf-8 -*-
# sequence.py
# author : Antoine Passemiers

from wynona.prot.sequence import Sequence
from wynona.prot.nw import NeedlemanWunsch


def align(fixed_length_seq, aligned_seq):
    fixed_length_seq = str(fixed_length_seq)
    aligned_seq = str(aligned_seq)
    if len(fixed_length_seq) != len(aligned_seq):
        seq_a = Sequence(fixed_length_seq)
        seq_b = Sequence(aligned_seq)

        alseq1, alseq2 = NeedlemanWunsch().align(seq_a, seq_b)
        alseq1, alseq2 = alseq1.__str__(), alseq2.__str__()

        indices = [i for i, res_name in enumerate(alseq1) if res_name != '-']
        aligned_seq = ''.join([alseq2[i] for i in indices])
    assert(len(fixed_length_seq) == len(aligned_seq))
    return aligned_seq


def align_to_itself(query_seq, whole_seq):
    if len(query_seq) != len(whole_seq):
        if not isinstance(query_seq, Sequence):
            query_seq = Sequence(query_seq)
        if not isinstance(whole_seq, Sequence):
            whole_seq = Sequence(whole_seq)
        alseq1, alseq2 = NeedlemanWunsch().align(query_seq, whole_seq)
        alseq1, alseq2 = alseq1.__str__(), alseq2.__str__()
        assert(len(whole_seq) == len(alseq2))
        indices = [i for i, res_name in enumerate(alseq1) if res_name != '-']
    else:
        indices = [i for i, res_name in enumerate(query_seq) if res_name != '-']
    return indices
