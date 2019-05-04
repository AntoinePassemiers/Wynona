from gaussfold import GaussFold, Optimizer, tm_score, rmsd
from gaussfold import PDBParser, SS3Parser, FastaParser, ContactParser

import os
import warnings
import numpy as np
from sklearn.manifold import MDS

with warnings.catch_warnings():
    warnings.simplefilter('ignore')


folders = os.listdir('PSICOV150')
folders = [folders[2]]
for folder in folders:

    sequence_name = folder.replace('_', '')
    DATA_FOLDER = os.path.join('PSICOV150', folder)

    #print(folder)

    filepath = os.path.join(DATA_FOLDER, 'sequence.fa')
    sequence = FastaParser().parse(filepath)['sequences'][0]

    filepath = os.path.join(DATA_FOLDER, 'ss3.txt')
    ssp = SS3Parser().parse(filepath).argmax(axis=1)

    L = len(sequence) # Number of residues
    filepath = os.path.join(DATA_FOLDER, 'closest.Jan13.contacts')
    distances = ContactParser(L, verbose=False).parse(filepath)

    valid_indices = np.where(np.sum(np.isnan(distances), axis=1) < L - 1)[0]
    nonan_distances = distances[valid_indices].T[valid_indices]
    
    embedding = MDS(
            n_components=3,
            metric=True,
            n_init=100,
            max_iter=300,
            eps=1e-4,
            dissimilarity='precomputed')
    points = embedding.fit_transform(nonan_distances)
    coords_target = [None] * L
    for i, point in zip(valid_indices, points):
        coords_target[i] = point


    filepath = os.path.join(DATA_FOLDER, 'sequence.fa.psicov2')
    cmap = ContactParser(L, target_cols=[4], verbose=False).parse(filepath)

    assert(not np.isnan(ssp).any())

    gf = GaussFold(sep=1, n_init_sols=1)
    gf.optimizer = Optimizer(
        pop_size=1000,        # Population size
        n_iter=200000,        # Maximum number of iterations
        partition_size=20,    # Partition size for the selection of parents
        mutation_rate=0.5,    # Percentage of child's points to be mutated
        mutation_std=.1,      # Stdv of mutation noise
        init_std=30.,         # Stdv for randomly generating initial solutions
        early_stopping=20000) # Maximum number of iterations without improvement
    coords_predicted = gf.run(cmap, ssp, verbose=True)

    tm = tm_score(coords_predicted, coords_target)
    r = rmsd(coords_predicted, coords_target)
    print('%s & %.2f & %.2f' % (sequence_name, tm, r))
