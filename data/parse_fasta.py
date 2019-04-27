# -*- coding: utf-8 -*-
# trim_alignments.py
# author : Antoine Passemiers

import os


def parse_fasta(filepath):
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