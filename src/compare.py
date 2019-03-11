# -*- coding: utf-8 -*-
# compare.py
# author : Antoine Passemiers

from deepcp.prot.contact_map import ContactMap
from deepcp.prot.data_manager import DataManager
from deepcp.prot.evaluation import Evaluation
from deepcp.prot.exceptions import ContactMapException
from deepcp.prot.utils import *

import os
import random
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = '../data'
TEMP_PATH = '../out/temp'

min_aa_separation = 6
target_contact_threshold = 8.


def create_latex_tabular(col_format, col_names, row_names, values):
    col_format = col_format
    col_names = col_names
    text = '\\begin{tabular}{%s}\n\t\\hline\n' % col_format
    text += '\t& ' + ' & '.join(col_names) + ' \\\\\n'
    text += '\t\\hline\n\t\\hline\n'

    for i, row_name in enumerate(row_names):
        row_values = ['%.2f' % float(v) for v in values[i]]
        text += '\t' + row_names[i] + ' & ' + ' & '.join(row_values) + ' \\\\\n'
        text += '\t\\hline\n'

    text += '\\end{tabular}\n'
    return text


if __name__ == '__main__':
    feature_indices = [1, 2, 3, 7, 8, 9, 10, 11]
    evaluation = Evaluation()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    for feature_set in data_manager.proteins(dataset='debug_val'):
        seq_name = feature_set.prot_name
        distances = feature_set.distances
        target_cmap = ContactMap(np.asarray(distances < target_contact_threshold, dtype=np.double))
        feature_names = feature_set.features['2-dim']['names']
        assert(feature_names[11] == 'metapsicov')
        features = np.asarray(feature_set.features['2-dim']['values'])
 
        for i in feature_indices:
            pred_cmap = ContactMap(features[i, :, :])
            name = feature_names[i]
            evaluation.add(name, seq_name, pred_cmap, target_cmap, min_aa_separation)

    row_names = [feature_names[i] for i in feature_indices]
    col_names = ['L', 'L/2', 'L/5']
    col_format = '|l|c|c|'
    scores = np.empty((len(row_names), len(col_names)), dtype=np.float)

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            scores[i, j] = evaluation.get(row_names[i], 'PPV', criterion=col_names[j])

    text = create_latex_tabular(col_format, col_names, row_names, scores)
    print(text)