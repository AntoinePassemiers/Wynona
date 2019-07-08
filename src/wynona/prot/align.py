# -*- coding: utf-8 -*-
# sequence.py
# author : Antoine Passemiers

import minineedle # https://github.com/scastlara/minineedle
import miniseq # https://github.com/scastlara/miniseq


def align(fixed_length_seq, aligned_seq):
    fixed_length_seq = str(fixed_length_seq)
    aligned_seq = str(aligned_seq)
    if len(fixed_length_seq) != len(aligned_seq):
        seq_a = miniseq.Protein('', fixed_length_seq)
        seq_b = miniseq.Protein('', aligned_seq)
        alignment = minineedle.Needleman(seq_a, seq_b)
        alignment.align()

        indices = [i for i, res_name in enumerate(alignment.alseq1) if res_name != '-']
        aligned_seq = ''.join([alignment.alseq2[i] for i in indices])
    assert(len(fixed_length_seq) == len(aligned_seq))
    return aligned_seq


def align_to_itself(self, query_seq, whole_seq):
    fixed_length_seq = str(fixed_length_seq)
    aligned_seq = str(aligned_seq)
    if len(query_seq) != len(whole_seq):
        query_seq = miniseq.Protein('', query_seq)
        whole_seq = miniseq.Protein('', whole_seq)
        alignment = minineedle.Needleman(query_seq, whole_seq)
        alignment.align()
        assert(len(whole_seq) == len(alignment.alseq2))
        indices = [i for i, res_name in enumerate(alignment.alseq1) if res_name != '-']
    else:
        indices = [i for i, res_name in enumerate(query_seq) if res_name != '-']
    return indices
