# -*- coding: utf-8 -*-
# compare.py
# author : Antoine Passemiers

from wynona.prot.contact_map import ContactMap
from wynona.prot.data_manager import DataManager
from wynona.prot.evaluation import Evaluation
from wynona.prot.exceptions import ContactMapException
from wynona.prot.utils import *

import os
import random
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = '../data'
TEMP_PATH = '../out/temp'

min_aa_separation = 6
target_contact_threshold = 8.




if __name__ == '__main__':
    evaluation = Evaluation()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    print('Name, N, Meff, PPV, PPV/2, PPV/5, PPV/10, PPV-short, PPV/2-short, PPV/5-short, PPV/10-short, PPV-medium, PPV/2-medium, PPV/5-medium, PPV/10-medium, PPV-long, PPV/2-long, PPV/5-long, PPV/10-long')
    for feature_set in data_manager.proteins(dataset='debug'):
        seq_name = feature_set.prot_name
        distances = feature_set.distances
        target_cmap = ContactMap(np.asarray(distances < target_contact_threshold, dtype=np.double))
        feature_names = feature_set.features['2-dim']['names']
        assert(feature_names[2] == 'plmdca')

        pred_cmap = feature_set.features['2-dim']['values'][2]

        pred_cmap = ContactMap(pred_cmap)
        Meff = np.sum(feature_set.msa_weights)
        evaluation.add('-', seq_name, pred_cmap, target_cmap, min_aa_separation)


        print('%s, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f' % (
            seq_name,
            len(pred_cmap.asarray()),
            Meff,
            evaluation.get('-', 'PPV', criterion='L', seq_name=seq_name),
            evaluation.get('-', 'PPV', criterion='L/2', seq_name=seq_name),
            evaluation.get('-', 'PPV', criterion='L/5', seq_name=seq_name),
            evaluation.get('-', 'PPV', criterion='L/10', seq_name=seq_name),
            evaluation.get('-', 'PPV-short', criterion='L', seq_name=seq_name),
            evaluation.get('-', 'PPV-short', criterion='L/2', seq_name=seq_name),
            evaluation.get('-', 'PPV-short', criterion='L/5', seq_name=seq_name),
            evaluation.get('-', 'PPV-short', criterion='L/10', seq_name=seq_name),
            evaluation.get('-', 'PPV-medium', criterion='L', seq_name=seq_name),
            evaluation.get('-', 'PPV-medium', criterion='L/2', seq_name=seq_name),
            evaluation.get('-', 'PPV-medium', criterion='L/5', seq_name=seq_name),
            evaluation.get('-', 'PPV-medium', criterion='L/10', seq_name=seq_name),
            evaluation.get('-', 'PPV-long', criterion='L', seq_name=seq_name),
            evaluation.get('-', 'PPV-long', criterion='L/2', seq_name=seq_name),
            evaluation.get('-', 'PPV-long', criterion='L/5', seq_name=seq_name),
            evaluation.get('-', 'PPV-long', criterion='L/10', seq_name=seq_name),
        ))
