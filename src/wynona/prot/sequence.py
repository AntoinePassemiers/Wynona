# -*- coding: utf-8 -*-
# sequence.py
# author : Antoine Passemiers

import numpy as np


AA_DTYPE = np.int8

N_STATES = 22
N_AA_SYMBOLS = 22
AA_SYMBOLS = [
    '-', # (Gap)
    'A', # Alanine
    'R', # Arginine
    'N', # Asparagine
    'D', # Aspartic acid
    'C', # Cysteine
    'E', # Glumatic acid
    'Q', # Glutamine
    'G', # Glycine
    'H', # Histidine
    'I', # Isoleucine
    'L', # Leucine
    'K', # Lysine
    'M', # Methionine
    'F', # Phenylalanine
    'P', # Proline
    'S', # Serine
    'T', # Threonine
    'W', # Tryptophan
    'Y', # Tyrosine
    'V', # Valine
    'X', # (Unknown)
    
    'B', # Asparagine / aspartic acid
    'U', # Selenocysteine (stop codon)
    'O', # Pyrrolysine (stop codon)
    'Z'] # Glutamine / glutamic acid

AA_CHAR_TO_INT = {key : i for i, key in enumerate(AA_SYMBOLS)}
AA_CHAR_TO_INT['Z'] = AA_CHAR_TO_INT['E']
AA_CHAR_TO_INT['O'] = AA_CHAR_TO_INT['K']
AA_CHAR_TO_INT['U'] = AA_CHAR_TO_INT['C']
AA_CHAR_TO_INT['B'] = AA_CHAR_TO_INT['D']


class Sequence:
    """Data structure for storing a sequence of amino acids. 
    The latter is represented by a contiguous array of integers.
    The mapping between the amino acids and their numeric value
    is done by checking in a dictionary
    
    Parameters:
        data (np.ndarray):
            contiguous array containing the symbols of the amino acids
        comments (list):
            List of informations about the sequence
            parsed from the FASTA file.
            The list is constructed by splitting the comments
            using the space delimiter.
    """

    def __init__(self, data, comments=list()):
        self.comments = comments
        if isinstance(data, np.ndarray):
            self.data = data.astype(AA_DTYPE)
        else:
            # If a string is passed, the latter is converted to a numpy array
            self.data = np.empty(len(data), dtype=AA_DTYPE)
            for i in range(len(data)):
                self.data[i] = Sequence.char_to_int(data[i])
    
    @staticmethod
    def random(L):
        """Generate a uniformly random sequence.

        Parameters:
            L (int):
                Sequence length.
        
        Returns:
            :obj:`Sequence`:
                Random sequence.
        """
        return Sequence(np.random.randint(0, N_AA_SYMBOLS, size=L))

    @staticmethod
    def char_to_int(c):
        """Converts an amino acid symbol to an integer that can be used
        to index data structures storing amino acid data.

        Parameters:
            c (str):
                Character representing an amino acid
        
        Returns:
            int:
                Integer i such that 0 <= i < 21
        """
        i = AA_CHAR_TO_INT[c.upper()]
        if not i < N_AA_SYMBOLS:
            print(i, c, c.upper())
        assert(i < N_AA_SYMBOLS)
        return i

    @staticmethod
    def int_to_char(i):
        """Converts an integer to an amino acid symbol.

        Parameters:
            i (int):
                Integer such that 0 <= i < 21
        
        Returns:
            str:
                Amino acid symbol
        """
        return AA_SYMBOLS[i]

    def trim(self, start, end):
        """Trim sequence by keeping only amino acids with positions
        in given range.

        Parameters:
            start (int):
                Start index of the amino acids to keep
            end (int):
                End index (not included) of the amino acids to keep

        Returns:
            :obj:`Sequence`:
                Trimmed sequence
        """
        return Sequence(self.data[start:end], comments=self.comments)

    def compute_identity(self, other):
        """The identity rate between the two sequences (self and other) 
        is determined by the number of a.a. pairs that are identical 
        and the total number of AA pairs 
        The function assumes that the two sequences have the same length.

        Parameters:
            other (:obj:`Sequence`):
                Sequence to be compared with
        
        Returns:
            float:
                Identity rate between the two sequence. Identity rate
                is comprised between zero and one
        """
        assert(self.__len__() == other.__len__())
        return (self.__len__() - np.count_nonzero(self.data[:] - other.data[:])) / self.__len__()

    def gap_fraction(self):
        """The fraction of gaps in the amino acid sequence.

        Returns:
            float:
                Gap fraction (a number in [0, 1])
        """
        return (self.data == 0).sum() / float(self.__len__())

    def to_array(self, one_hot_encoded=False, states=False):
        """Represents the sequence as an array of integers. """
        seq = np.asarray(self.data, dtype=np.uint8)
        # TODO
        n_symbols = N_AA_SYMBOLS if not states else N_STATES
        if one_hot_encoded:
            data = np.zeros((self.__len__(), n_symbols), dtype=np.uint8)
            data[np.arange(self.__len__()), seq] = 1
            seq = data
        return seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Created a new sequence containing the requested subset
            return Sequence(self.data[key], self.comments)
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(value, Sequence):
            # Assigning multiple values at once
            self.data[key] = value.data[:]
        else:
            self.data[key] = value

    def __str__(self):
        """Convert sequence back to its string representation """
        s = ''
        for i in range(self.__len__()):
            s += Sequence.int_to_char(self.data[i])
        return s

    def __repr__(self):
        """Return information/comments about the sequence """
        return self.__str__()

    def __eq__(self, other):
        """Check whether two sequences are perfectly identical """
        if isinstance(other, str):
            other = Sequence(other)
        return (self.__len__() == other.__len__()) and \
            (np.count_nonzero(self.data[:] - other.data[:]) == 0)
