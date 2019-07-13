# -*- coding: utf-8 -*-
# parsers.py
# author : Antoine Passemiers

from wynona.prot.sequence import Sequence
from wynona.prot.feature_set import FeatureSet
from wynona.prot.utils import *
from wynona.prot.exceptions import UnsupportedExtensionError
from wynona.prot.features import *
from wynona.prot.cov import *

import os
import warnings
import numpy as np
import pandas as pd

try:
    # Subclassing is not required
    # -> We can use pickle's C implementation
    import cPickle as pickle
except:
    import pickle


class Parser:

    def getSupportedExtensions(self):
        return list()

    def parse(self, filepath):
        _, file_ext = os.path.splitext(filepath)
        if not file_ext in self.getSupportedExtensions():
            raise UnsupportedExtensionError(
                "Extension %s is not supported by parser %s" % (file_ext, self.__class__.__name__))
        else:
            return self.__parse__(filepath)


class PairsParser(Parser):

    def getSupportedExtensions(self):
        return list()

    def __parse_pairs__(self, filepath, delimiter=',', target_col=2, column_names=list(), sequence_length=None):
        assert("target" in column_names)
        with open(filepath, "r") as f:
            lines = f.readlines()
            try:
                if sequence_length is None:
                    dataframe = pd.read_csv(filepath, sep=delimiter, skip_blank_lines=True,
                        header=None, names=column_names, index_col=False)
                    sequence_length = np.asarray(dataframe[["i", "j"]]).max()
            except ValueError:
                return None
            data = np.full((sequence_length, sequence_length), np.nan, dtype=np.double)
            np.fill_diagonal(data, 0)
            for line in lines:
                elements = line.rstrip("\r\n").split(delimiter)
                i, j, k = int(elements[0]) - 1, int(elements[1]) - 1, float(elements[target_col])
                data[i, j] = data[j, i] = k
            if np.isnan(data).any():
                # sequence_length is wrong or the input file has missing pairs
                warnings.warn("Warning: Pairs of residues are missing from the contacts text file")
                warnings.warn("Number of missing pairs: %i " % np.isnan(data).sum())
            return data


class SS3Parser(Parser):

    def __init__(self, target_indices=[3, 4, 5]):
        self.target_indices = target_indices

    def getSupportedExtensions(self):
        return ['.ss3', '.acc', '.diso', '.txt']

    def __parse__(self, filepath):
        data = list()
        with open(filepath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                pass
            elif len(line) > 2:
                els = line.split()
                data.append([float(els[i]) for i in self.target_indices])
        return np.asarray(data)


class CBParser(PairsParser):

    def getSupportedExtensions(self):
        return [".contacts", ".CB"]

    def __parse__(self, filepath):
        return self.__parse_pairs__(filepath, delimiter=' ', target_col=2,
            column_names = ["i", "j", "target"])


class CCMPredFileParser(Parser):

    def __init__(self, sequence_length, delimiter=' '):
        self.sequence_length = sequence_length
        self.delimiter = delimiter

    def getSupportedExtensions(self):
        return ['.out']

    def __parse__(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            data = np.full((1, self.sequence_length, self.sequence_length), np.nan, dtype=np.double)
            np.fill_diagonal(data[0, :, :], 0)
            i = 0
            for line in lines:
                elements = line.rstrip('\r\n').split()
                if len(elements) == data.shape[1]:
                    for j in range(data.shape[1]):
                        data[0, i, j] = float(elements[j])
                    i += 1
            #assert(i == self.sequence_length)
            return data


class PredictionFileParser(Parser):

    def __init__(self, sequence_length, delimiter=' ', target_cols=None):
        self.sequence_length = sequence_length
        self.delimiter = delimiter
        self.target_cols = target_cols

    def getSupportedExtensions(self):
        return ['.out', '.gaussdca', '.psicov2', '.plmdca2']

    def is_comment(self, elements):
        is_comment = False
        for element in elements:
            try:
                float(element)
            except ValueError:
                is_comment = True
                break
        return is_comment

    def get_features(self, elements):
        features = list()
        if self.target_cols:
            for i in self.target_cols:
                features.append(float(elements[i]))
        else:
            target_col = len(elements) - 1
            features.append(float(elements[target_col]))
        return features

    def __parse__(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            n_features = 1 if not self.target_cols else len(self.target_cols)
            data = np.full((n_features, self.sequence_length, self.sequence_length), np.nan, dtype=np.double)
            data[:, np.arange(self.sequence_length), np.arange(self.sequence_length)] = 0
            for line in lines:
                if self.delimiter:
                    elements = line.rstrip('\r\n').split(self.delimiter)
                else:
                    elements = line.rstrip('\r\n').split()
                if not self.is_comment(elements):
                    i, j = int(elements[0]) - 1, int(elements[1]) - 1
                    data[:, i, j] = data[:, j, i] = self.get_features(elements)
            if np.isnan(data).any():
                # sequence_length is wrong or the input file has missing pairs
                warnings.warn('Warning: Pairs of residues are missing from the contacts text file')
                warnings.warn('Number of missing pairs: %i ' % np.isnan(data).sum())
            return data


class FastaParser(Parser):

    def getSupportedExtensions(self):
        return ['.fasta', '.fa', '.a3m', '.trimmed']

    def __parse__(self, filepath):
        with open(filepath, 'r') as f:
            data = { 'sequences' : list(), 'comments' : list() }
            lines = f.readlines()
            sequence = ""
            for line in lines:
                if line[0] == '>':
                    data['comments'].append(line[1:].rstrip("\r\n"))
                    if len(sequence) > 0:
                        sequence = Sequence(sequence.rstrip("\r\n"))
                        data['sequences'].append(sequence)
                        sequence = ''
                else:
                    line = line.replace('\n', '').replace('\r', '').strip()
                    if len(line) > 1:
                        sequence += line
            sequence = Sequence(sequence.rstrip("\r\n"))
            data['sequences'].append(sequence)
            assert(len(data['sequences']) == len(data['comments']))
            return data


def parse_folder(folder, prot_name):
    # Retrieve sequence and sequence length
    fa = FastaParser().parse(os.path.join(folder, 'sequence.fa'))
    sequence = fa['sequences'][0]
    assert(isinstance(sequence, Sequence))
    L = len(sequence)

    # Get Distance map
    if os.path.isfile(os.path.join(folder, 'closest.Jan13.contacts')):
        distances = np.squeeze(CBParser().parse(os.path.join(folder, 'closest.Jan13.contacts')))
        coordinates = None
    elif os.path.isfile(os.path.join(folder, 'contacts.CB')):
        distances = np.squeeze(CBParser().parse(os.path.join(folder, 'contacts.CB')))
        coordinates = None
    else:
        distances, coordinates = PDBParser(sequence, prot_name).parse(os.path.join(folder, 'native.pdb'))
    print((np.isnan(distances).sum() / float(L ** 2)))

    # Get Multiple Sequence Alignment for given sequence
    if os.path.isfile(os.path.join(folder, 'trimmed.a3m')):
        alignment = FastaParser().parse(os.path.join(folder, 'trimmed.a3m'))['sequences']
    else:
        alignment = FastaParser().parse(os.path.join(folder, 'sequence.fa.blits4.trimmed'))['sequences']
    msa = np.asarray([sequence.to_array() for sequence in alignment], dtype=np.uint8)
    #msa_weights = compute_weights(msa, 0.8)
    msa_weights = np.ones(len(msa))

    # RaptorX-property
    ss3 = SS3Parser([3, 4, 5]).parse(os.path.join(folder, 'ss3.txt')).T
    acc = SS3Parser([3, 4, 5]).parse(os.path.join(folder, 'acc.txt')).T
    diso = SS3Parser([3]).parse(os.path.join(folder, 'diso.txt')).T

    # Create feature set to be saved later
    features = FeatureSet(prot_name, alignment, msa_weights, distances, coordinates, ss3.argmax(axis=0))

    # Add GaussDCA and plmDCA predictions
    gdca_dir = PredictionFileParser(L).parse(os.path.join(folder, 'dir.gaussdca'))
    features.add('gdca-dir', gdca_dir)
    gdca_fnr = PredictionFileParser(L).parse(os.path.join(folder, 'fnr.gaussdca'))
    features.add('gdca-fnr', gdca_fnr)
    if os.path.isfile(os.path.join(folder, 'plmdca.out')):
        plmdca = PredictionFileParser(L, delimiter=',').parse(os.path.join(folder, 'plmdca.out'))
    else:
        plmdca = PredictionFileParser(L, delimiter=',').parse(os.path.join(folder, 'sequence.fa.plmdca2'))
    features.add('plmdca', plmdca)
    if os.path.isfile(os.path.join(folder, 'psicov.out')):
        psicov = PredictionFileParser(L).parse(os.path.join(folder, 'psicov.out'))
    else:
        psicov = PredictionFileParser(L).parse(os.path.join(folder, 'sequence.fa.psicov2'))
    features.add('psicov', psicov)

    # Get additional features from multiple sequence alignment
    feature_names = [
        'mutual-information',
        'normalized-mutual-information',
        'cross-entropy']
    mi, nmi, cross_entropy = extract_features_2d(msa, feature_names)
    features.add('mutual-information', mi)
    features.add('normalized-mutual-information', nmi)
    features.add('cross-entropy', cross_entropy)

    # Add PhyCMAP predictions
    if os.path.isfile(os.path.join(folder, 'phycmap.out')):
        phycmap = PredictionFileParser(L).parse(os.path.join(folder, 'phycmap.out'))
        features.add('phycmap', phycmap)

    # Add CCMPred predictions if available
    if os.path.isfile(os.path.join(folder, 'ccmpred.out')):
        ccmpred = CCMPredFileParser(L).parse(
                os.path.join(folder, 'ccmpred.out'))
        features.add('ccmpred', ccmpred)

    # Add EVfold predictions if available
    if os.path.isfile(os.path.join(folder, 'evfold.out')):
        evfold = PredictionFileParser(L, target_cols=[4, 5]).parse(
                os.path.join(folder, 'evfold.out'))
        features.add('evfold', evfold)

    # Add MetaPSICOV predictions if available
    if os.path.isfile(os.path.join(folder, 'metapsicov.stage2.out')):
        metapsicov = PredictionFileParser(L).parse(
                os.path.join(folder, 'metapsicov.stage2.out'))
        features.add('metapsicov', metapsicov)

    # Add PConsC3 predictions if available
    if os.path.isfile(os.path.join(folder, 'pconsc3.l5.out')):
        pconsc3 = PredictionFileParser(L).parse(
                os.path.join(folder, 'pconsc3.l5.out'))
        features.add('pconsc3', pconsc3)

    # Get 1D features
    self_information, partial_entropy = extract_features_1d(msa, ['self-information', 'partial-entropy'])
    ohe_sequence = sequence.to_array(one_hot_encoded=True)
    features.add('self-information', self_information.T)
    features.add('partial-entropy', partial_entropy.T)
    features.add('ohe-sequence', ohe_sequence.T)
    assert(diso.shape[0] == 1)
    features.add('ss3', ss3)
    features.add('acc', acc)
    features.add('diso', diso)

    # Add global features
    features.add('sequence-length', L)
    #M_eff = np.sum(msa_weights)
    #features.add('M-eff', M_eff)

    return features.concat()
