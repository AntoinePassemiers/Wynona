# -*- coding: utf-8 -*-
# trim_alignments.py
# author : Antoine Passemiers

from parse_fasta import parse_fasta

import os


def write_fasta(filepath, data):
    with open(filepath, 'w') as f:
        for i in range(len(data['sequences'])):
            f.write('>' + data['comments'][i] + '\n')
            f.write(data['sequences'][i] + '\n')


def trim(alignment):
    L = len(alignment[0])
    for i in range(1, len(alignment)):
        alignment[i] = alignment[i][:L]



if __name__ == '__main__':
    for dp, dn, filenames in os.walk('.'):
        for f in filenames:
            if f == 'alignment.a3m':
                print(os.path.join(dp, f))
                data = parse_fasta(os.path.join(dp, f))
                trim(data['sequences'])
                assert(len(data['sequences']) == len(data['comments']))
                write_fasta(os.path.join(dp, f), data)
