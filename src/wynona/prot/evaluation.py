# -*- coding: utf-8 -*-
# evaluation.py
# author : Antoine Passemiers

from wynona.prot.exceptions import ContactMapException, EvaluationException

import copy
import numpy as np


class Evaluation:

    class Entry:

        def __init__(self, name, pred_cmap, target_cmap, min_aa_separation, criterion='L'):
            self.name = name
            self.criterion = criterion
            self.L = len(pred_cmap)
            if criterion == 'L':
                n_top = self.L
            elif criterion == 'L/2':
                n_top = self.L // 2
            elif criterion == 'L/5':
                n_top = self.L // 5
            elif criterion == 'L/10':
                n_top = self.L // 10
            else:
                raise EvaluationException('Unknown evaluation criterion: %s' % criterion)
            self.pred_cmap = pred_cmap
            self.target_cmap = target_cmap
            self.min_aa_separation = min_aa_separation
            self.metrics = self.compute_metrics(n_top)
            for lb, ub, name in [(min_aa_separation, 12, 'short'),
                                 (12, 24, 'medium'),
                                 (24, None, 'long')]:
                new_metrics = self.compute_metrics(n_top, _range=(lb, ub))
                for key in new_metrics.keys():
                    self.metrics[key+'-'+name] = new_metrics[key]
        
        def compute_metrics(self, n_top, _range=[None, None]):
            #assert(self.pred_cmap.is_binary() and self.target_cmap.is_binary())
            if _range[0] is None:
                _range[0] = self.min_aa_separation
            pred_cmap = self.pred_cmap.copy()
            predictions = pred_cmap.in_range(_range[0], _range[1])
            try:
                predictions = predictions.top(n_top)
            except ContactMapException:
                pass
            targets = self.target_cmap.in_range(_range[0], _range[1])
            nonan = ~np.isnan(targets)
            predictions = predictions[nonan]
            targets = targets[nonan]
            assert(np.logical_or(predictions == True, predictions == False).all())
            assert(np.logical_or(targets == True, targets == False).all())
            TP = np.logical_and(predictions == True, targets == True).sum()
            FP = np.logical_and(predictions == True, targets == False).sum()
            FN = np.logical_and(predictions == False, targets == True).sum()
            TN = np.logical_and(predictions == False, targets == False).sum()
            N = TP + FP + FN + TN
            PPV = float(TP) / float(TP + FP) if TP != 0. else 0.
            FPV = float(TN) / float(TN + FN) if TN != 0. else 0.
            ACC = float(TP + TN) / float(N)
            TPR = float(TP) / float(TP + FN) if TP != 0. else 0.
            FPR = float(FP) / float(FP + TN) if FP != 0. else 0.
            tp, tn, fp, fn = float(TP) / N, float(TN) / N, float(FP) / N, float(FN) / N
            F1 = 2 * TP / float(2 * TP + FP + FN)
            metrics = {
                'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'N': N, 'PPV': PPV,
                'FPV': FPV, 'ACC': ACC, 'F1': F1 }
            return metrics
        
        def keys(self):
            return self.metrics.keys()

        def __getitem__(self, key):
            return self.metrics[key]

    def __init__(self):
        self.data = dict()
    
    def dir_name(self, algorithm_name, criterion):
        return algorithm_name + criterion
    
    def add(self, algorithm_name, seq_name, pred_cmap, target_cmap, min_aa_separation):
        for criterion in ['L/10', 'L/5', 'L/2', 'L']:
            entry = Evaluation.Entry(
                seq_name, pred_cmap, target_cmap, min_aa_separation, criterion=criterion)
            name = self.dir_name(algorithm_name, criterion)
            if not name in self.data.keys():
                self.data[name] = {
                    'entries': dict(),
                    'averaged-metrics': dict() }
            self.data[name]['entries'][seq_name] = entry
            self.data[name]['averaged-metrics'] = dict()
            n_entries = len(self.data[name]['entries'].values())
            for metric_name in entry.keys():
                for entry in self.data[name]['entries'].values():
                    if metric_name not in self.data[name]['averaged-metrics'].keys():
                        self.data[name]['averaged-metrics'][metric_name] = entry[metric_name]
                    else:
                        self.data[name]['averaged-metrics'][metric_name] += entry[metric_name]
                self.data[name]['averaged-metrics'][metric_name] /= n_entries

    def get(self, algorithm_name, metric_name, criterion='L', seq_name=None):
        name = self.dir_name(algorithm_name, criterion)
        if seq_name is None:
            return self.data[name]['averaged-metrics'][metric_name]
        else:
            return self.data[name]['entries'][seq_name][metric_name]
