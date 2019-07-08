# -*- coding: utf-8 -*-
# pdb_parser.py
# author : Antoine Passemiers

from wynona.prot.align import align_to_itself
from wynona.prot.parsers import Parser

import numpy as np


RES_NAME_TO_SYM = {
    'ALA': 'A',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'CYS': 'C',
    'GLU': 'E',
    'GLN': 'Q',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'MET': 'M',
    'MSE': 'M',
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'THR': 'T',
    'TRP': 'W',
    'TYR': 'Y',
    'VAL': 'V'
}


class ResidueId:

    def __init__(self, idf):
        self.idf = idf.strip()
        self.number = ''
        self.suffix = ''
        for c in self.idf:
            if c.isdigit() or c in ['+', '-']:
                self.number += c
            else:
                self.suffix += c
        self.number = int(self.number)
        assert(len(self.suffix) <= 1)

    def __float__(self):
        f = self.number * 1000
        if self.suffix:
            f += ord(self.suffix)
        return f

    def __le__(self, other):
        return self.__float__() <= other.__float__()

    def __ge__(self, other):
        return self.__float__() >= other.__float__()

    def __eq__(self, other):
        return self.__float__() == other.__float__()

    def __ne__(self, other):
        return self.__float__() != other.__float__()

    def __lt__(self, other):
        return self.__float__() < other.__float__()

    def __gt__(self, other):
        return self.__float__() > other.__float__()
    
    def __hash__(self):
        return hash(self.__float__())


class PDBParser(Parser):

    def __init__(self, sequence, prot_name, method='CASP'):
        self.sequence = sequence
        self.L = len(sequence)
        self.prot_name = prot_name
        self.method = method

    def getSupportedExtensions(self):
        return ['.pdb']

    def isAppropriateAtom(self, res_name, atom):
        valid = False
        if self.method == 'CA' and atom == 'CA':
            valid = True
        elif self.method == 'CB' and atom == 'CB':
            valid = True
        elif self.method == 'CASP':
            if res_name == 'GLY' and atom == 'CA':
                valid = True
            elif res_name != 'GLY' and atom == 'CB':
                valid = True
        return valid

    def res_name_to_sym(self, res_name):
        try:
            sym = RES_NAME_TO_SYM[res_name]
        except KeyError:
            sym = 'X' # TODO
        return sym

    def __parse__(self, filepath):
        if self.prot_name[0] == 'T' and '-' in self.prot_name:
            het_chain = None
        else:
            het_chain = self.prot_name[-1]
        distances = np.full((self.L, self.L), np.nan, dtype=np.float)
        
        residues = dict()
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line[:10] == 'REMARK 465' and line[19] == het_chain:
                res_name = line[15:19].strip()
                res_id = line[21:27].strip()
                if any(c.isdigit() for c in res_id):
                    res_id = ResidueId(res_id)
                    residues[res_id] = (res_name, None)

        for line in lines:
            if (line[:4] == 'ATOM' or line[:6] == 'HETATM') and (line[21] == het_chain or het_chain is None):
                res_id = ResidueId(line[23:27].strip())
                res_name = line[17:20].strip()
                atom = line[13:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if self.isAppropriateAtom(res_name, atom):
                    residues[res_id] = (res_name, np.asarray([x, y, z]))
                elif atom == 'CA' and not res_id in residues.keys():
                    residues[res_id] = (res_name, np.asarray([x, y, z]))

        residues = [x[1] for x in list(sorted(residues.items(), key=lambda kv: kv[0]))]
        whole_seq = ''.join([self.res_name_to_sym(pos[0]) for pos in residues])
        query_seq = str(self.sequence)
        print(whole_seq)
        print(query_seq)
       
        indices = align_to_itself(query_seq, whole_seq)
        coordinates = [residues[i][1] for i in indices]
        assert(len(coordinates) == self.L)

        for i, coords_i in enumerate(coordinates):
            for j, coords_j in enumerate(coordinates):
                if coords_i is not None and coords_j is not None:
                    distances[i, j] = np.sqrt(np.sum((coords_i - coords_j) ** 2.))
        return distances, coordinates
