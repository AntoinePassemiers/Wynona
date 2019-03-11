# -*- coding: utf-8 -*-
# hyperoptimizer.py
# author : Antoine Passemiers

import hyperopt
import os
import time
import json


class Hyperoptimizer:

    def __init__(self, space, train_func, save_folder, max_evals=5):
        self.train_func = train_func
        self.save_folder = save_folder
        self.max_evals = max_evals
        self.space = dict()
        for name in space:
            self.space[name] = hyperopt.hp.choice(name, space[name])
        self.trials = hyperopt.Trials()
        self.best = None
        self.current_id = 0

        def objective(params):
            start_time = time.time()
            result = self.train_func(params)
            self.save_result(params, result)
            return {
                'loss': result['loss'],
                'status': hyperopt.STATUS_OK,
                'eval_time': start_time,
                'computation_time': time.time() - start_time,
                'extra': result
            }
        self.objective = objective

    def run(self):
        self.reset()
        self.best = hyperopt.fmin(
            self.objective,
            self.space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials)

    def save_result(self, params, result):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        with open(os.path.join(self.save_folder, '%i.result' % self.current_id), 'w') as f:
            data = { 'params': params }
            for key in result.keys():
                if isinstance(result[key], (int, float, bool, list, dict)):
                    data[key] = result[key]
            f.write(json.dumps(data, indent=4, sort_keys=True))
            self.current_id += 1

    def reset(self):
        self.current_id = 0
        for subdir in os.listdir(self.save_folder):
            print(subdir)
            if subdir.endswith('.result'):
                os.remove(os.path.join(self.save_folder, subdir))
