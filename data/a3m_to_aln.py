# -*- coding: utf-8 -*-
# a3m_to_aln.py
# author : Antoine Passemiers

from parse_fasta import *


if __name__ == '__main__':
    for dp, dn, filenames in os.walk('.'):
        for f in filenames:
            if f == 'alignment.a3m':
                print(os.path.join(dp, f))
                data = parse_fasta(os.path.join(dp, f))
                trim(data['sequences'])
                assert(len(data['sequences']) == len(data['comments']))
                L = len(data['sequences'][0])
                for i, seq in enumerate(data['sequences']):
                    data['sequences'][i] = seq[:L]
                with open(os.path.join(dp, 'alignment.aln'), 'w') as out_file:
                    out_file.write('%s\n' % dp)
                    out_file.write('%i\n' % len(data['sequences']))
                    for seq in data['sequences']:
                        out_file.write('%s\n' % seq)
