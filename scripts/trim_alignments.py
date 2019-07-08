# -*- coding: utf-8 -*-
# trim_alignments.py
# author : Antoine Passemiers

from wynona.prot import align
from wynona.prot.parsers import FastaParser

import os


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '../data/training_set')


if __name__ == '__main__':

    for folder in os.listdir(DATA_FOLDER):

        filepath = os.path.join(DATA_FOLDER, folder, 'alignment.a3m')
        if os.path.exists(filepath):
            print('Processing folder %s' % folder)
            data = FastaParser().parse(filepath)

            filepath = os.path.join(DATA_FOLDER, folder, 'trimmed.a3m')
            with open(filepath, 'w') as f:
                target_sequence = str(data['sequences'][0])
                comment = data['comments'][0]
                f.write('>%s\n' % comment)
                f.write(str(target_sequence) + '\n')
                for i in range(1, len(data['comments'])):
                    sequence = str(data['sequences'][i])
                    comment = data['comments'][i]
                    sequence = align(target_sequence, sequence)
                    f.write('>%s\n' % comment)
                    f.write(str(sequence) + '\n')

