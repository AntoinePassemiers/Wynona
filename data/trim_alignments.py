# -*- coding: utf-8 -*-
# trim_alignments.py
# author : Antoine Passemiers

import os


def parse_fasta(filepath):
    print(filepath)
    with open(filepath, 'r') as f:
        data = { 'sequences' : list(), 'comments' : list() }
        lines = f.readlines()
        sequence = ""
        for i, line in enumerate(lines):
            if line[0] == '>':
                data['comments'].append(line[1:].rstrip("\r\n"))
                if len(sequence) > 0:
                    sequence = sequence.rstrip("\r\n")
                    data['sequences'].append(sequence.upper())
                    sequence = ''
            else:
                line = line.replace('\n', '').replace('\r', '').strip()
                if len(line) > 1:
                    sequence += line
        sequence = sequence.rstrip("\r\n")
        data['sequences'].append(sequence.upper())
        return data


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
