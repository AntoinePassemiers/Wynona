# -*- coding: utf-8 -*-
# contact_map.py
# author : Antoine Passemiers

from deepcp.prot.exceptions import ContactMapException

import numpy as np
import warnings
from scipy.linalg import toeplitz
import torch


class ContactMap:
    """Data structure representing a contact map.
    When elements are integer, the map is a predicted contact map.
    Otherwise, the map is either a thresholded
    predicted contact map or a ground-truth contact map.

    Attributes:
        cmap (:obj:`np.ndarray`):
            Map of shape (L, L), where L is the sequence length.
        L (int):
            Sequence length.
        shape (tuple):
            Shape of contact map is always (L x L).
    """

    def __init__(self, cmap):
        if isinstance(cmap, torch.Tensor):
            cmap = cmap.data.numpy()
        cmap = np.asarray(cmap)
        if cmap.ndim == 2:
            self.cmap = cmap
        elif cmap.ndim == 3:
            if cmap.shape[2] > 1:
                self.cmap = cmap[:, :, 1]
            else:
                self.cmap = cmap[:, :, 0]
        else:
            raise ContactMapException(
                'Incorrect shape for contact map: %s' % str(cmap.shape))
        self.L = self.cmap.shape[0]
        self.shape = (self.L, self.L)
    
    def is_binary(self):
        """Returns True when the map is either a thresholded predicted contact map
        or a ground-truth contact map.

        Returns:
            bool:
                Whether the contact map is binary or not.
        """
        return np.logical_or(self.cmap == 1, self.cmap == 0).all()
    
    def top(self, Lk, min_aa_separation=6):
        """Keep only Lk top predictions in lower triangle with given minimal
        residue distance.

        Parameters:
            Lk (int):
                Number of top predictions to retain.
            min_aa_separation (int):
                Minimum number of residue separation. All elements with
                residue distance lower than this threshold are discarded.

        Returns:
            :obj:`ContactMap`:
                New contact map where only Lk top predictions in lower triangle
                w.r.t. given residue distance are not set to zero.
        """
        if self.is_binary():
            raise ContactMapException('Calling top(...) on binary contact map')
        indices = np.tril_indices(self.L, k=-min_aa_separation-12)
        idx = np.argsort(self.cmap[indices[0], indices[1]])
        indices = (indices[0][idx[-Lk:]], indices[1][idx[-Lk:]])
        new_cmap = np.zeros((self.L, self.L), dtype=np.bool)
        new_cmap[indices] = 1
        return ContactMap(new_cmap)
    
    def in_range(self, lower_bound, upper_bound=None, symmetric=False, value=0):
        mask = np.zeros((self.L, self.L), dtype=np.bool)
        more_than_lb = np.tril_indices(self.L, -lower_bound)
        mask[more_than_lb] = True
        if upper_bound is not None:
            more_than_ub = np.tril_indices(self.L, -upper_bound)
            mask[more_than_ub] = False
        if symmetric:
            mask = np.logical_or(mask, mask.T)
        new_cmap = np.copy(self.cmap)
        new_cmap[~mask] = 0
        return ContactMap(new_cmap)

    @staticmethod
    def from_distances(distances, T=8, n=0):
        """Return a (L x L) boolean matrix where the (i, j) element tells
        whether the pair of residues (r(i), (j)) are involved in a contact.
        A contact respects the following condition: d(r(i),r(j)) <= T and |i-j| >= n

        Parameters:
            distances (:obj:`np.ndarray`):
                Distance matrix.
            T (float, or str):
                Distance threshold (cutoff), typically 8 Angstroms.
            n (int):
                Sequence separation parameter, optional (can default to 0).

        References:
            [1] http://genesilico.pl/gdserver/GDFuzz3D/theory.html
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            contacts = (distances <= T)
        contacts[np.isnan(contacts)] = True
        return ContactMap(np.asarray(
            np.logical_and(
                contacts, 
                toeplitz(
                    np.arange(distances.shape[0]),
                    np.arange(distances.shape[0])) >= n),
            dtype=np.uint8))

    def copy(self):
        return ContactMap(np.copy(self.cmap))
    
    def asarray(self):
        return self.cmap

    def __len__(self):
        return self.cmap.shape[0]
    
    def __setitem__(self, index, value):
        self.cmap[index] = value
    
    def __getitem__(self, index):
        return self.cmap[index]
    
    def get_rhs(self, other):
        if isinstance(other, ContactMap):
            return other.cmap
        else:
            return other
    
    def __eq__(self, other):
        return self.cmap == self.get_rhs(other)
    
    def __le__(self, other):
        return self.cmap <= self.get_rhs(other)
    
    def __lt__(self, other):
        return self.cmap < self.get_rhs(other)

    def __ge__(self, other):
        return self.cmap >= self.get_rhs(other)

    def __gt__(self, other):
        return self.cmap > self.get_rhs(other)
