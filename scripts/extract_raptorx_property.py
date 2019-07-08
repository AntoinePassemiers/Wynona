# -*- coding: utf-8 -*-
# extract_raptorx_property.py
# author : Antoine Passemiers

import os
from shutil import copyfile


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT, '../data')
PROPERTY_FOLDER = os.path.join(DATA_FOLDER, 'property')


if __name__ == '__main__':

    for folder in os.listdir(PROPERTY_FOLDER):
        folder_path = os.path.join(PROPERTY_FOLDER, folder)

        with open(os.path.join(folder_path, '%s.fasta.txt' % folder), 'r') as f:
            line = f.readline().replace('\n', '')
            assert(line[0] == '>')
            prot_name = line[1:]
            assert(len(prot_name) == 5)
            print('Processing folder %s' % prot_name)

        dest_path = os.path.join(DATA_FOLDER, 'training_set', prot_name)
        copyfile(os.path.join(folder_path, 'Profile_data', '%s.a3m' % folder),
                 os.path.join(dest_path, 'alignment.a3m'))
        for file_type in ['acc', 'diso', 'ss3']:
            copyfile(os.path.join(folder_path, '%s.%s.txt' % (folder, file_type)),
                     os.path.join(dest_path, '%s.txt' % file_type))
