# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from deepcp.prot.contact_map import ContactMap
from deepcp.prot.data_manager import DataManager
from deepcp.prot.evaluation import Evaluation
from deepcp.prot.exceptions import EarlyStoppingException
from deepcp.prot.utils import *
from deepcp.prot.parsers import apply_apc
from deepcp.nn import ConvNet, AdaptiveDataLoader, BinaryCrossEntropy
from deepcp.nn import Hyperoptimizer

import torch
from torch.autograd import Variable
import os
import random
import numpy as np
import matplotlib.pyplot as plt


DEBUG = True


DATA_PATH = '../data'
TEMP_PATH = '../out/temp'

min_aa_separation = 6
early_stopping = 1
num_epochs = 1 #TODO

HYPER_PARAM_SPACE = {
    'activation': ['relu'],
    'batch_size': [4],
    'bn_momentum': [1.0],
    'bn_track_running_stats': [False],
    'learning_rate': [5 * 1e-4],
    'l2_penalty': [1e-5],
    'momentum_decay_first_order': [0.5],
    'momentum_decay_second_order': [0.999],
    'use_batch_norm': [True]
}

contact_thresholds = [6., 7.5, 8., 8.5, 10.]
target_contact_threshold = 8.

n_0d_features = 2
n_1d_features = 114
n_2d_features = 4


if DEBUG:
    training_set_name = 'debug'
    validation_set_name = 'debug_val'
    max_evals = 1
else:
    training_set_name = 'training_set'
    validation_set_name = 'benchmark_set_membrane'
    max_evals = 1 # TODO


def feature_set_to_tensors(feature_set, remove_diag=False):
    distances = feature_set.distances
    L = distances.shape[1]

    contacts = list()
    for contact_threshold in contact_thresholds:
        cmap = ContactMap(distances < contact_threshold)
        if remove_diag:
            cmap = cmap.in_range(min_aa_separation, None, symmetric=True, value=np.nan)
        contacts.append(cmap.asarray())
    contacts = np.asarray(contacts, dtype=np.double)[np.newaxis, ...]

    Y = torch.as_tensor(contacts, dtype=torch.float32)

    X_0D = torch.as_tensor(np.asarray(feature_set.features['global']['values']), dtype=torch.float32)
    if X_0D.size()[0] >= 2:
        X_0D[0] /= 1000.
        X_0D[1] /= 10000.
    X_1D = torch.as_tensor(np.asarray(feature_set.features['1-dim']['values']), dtype=torch.float32)
    X_2D = torch.as_tensor(np.asarray(feature_set.features['2-dim']['values']), dtype=torch.float32)
    X_2D = X_2D[:n_2d_features, :, :]

    for i, x in enumerate(X_2D):
        X_2D[i, :, :] = torch.as_tensor(apply_apc(x))
    #X_2D[6, :, :] = torch.as_tensor(apply_apc(X_2D[6, :, :]))

    X = (X_0D, X_1D, X_2D)
    return X, Y


def train_model(data_manager, params, state_dict_path=None):
    print('Generating dataset...')
    training_set = list()
    for feature_set in data_manager.proteins(dataset=training_set_name):
        X, Y = feature_set_to_tensors(feature_set, remove_diag=False)
        (X_0D, X_1D, X_2D) = X
        if X_0D.size()[0] > 0 and X_1D.size()[0] > 0 and X_2D.size()[0]:
            training_set.append((X, Y))

    # Initialize data loader
    data_loader = AdaptiveDataLoader(
            dataset=training_set,
            shuffle=True,
            batch_size=params['batch_size'])

    # Initialize model
    model = ConvNet(
            n_0d_features,
            n_1d_features,
            n_2d_features,
            len(contact_thresholds),
            use_batch_norm=params['use_batch_norm'],
            bn_momentum=params['bn_momentum'],
            bn_track_running_stats=params['bn_track_running_stats'])
    model.init_weights()
    model.train() # Activate dropout, batchnorm, etc.

    # Initialize optimizer
    b1 = params['momentum_decay_first_order']
    b2 = params['momentum_decay_second_order']
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            betas=(b1, b2),
            weight_decay=params['l2_penalty'])

    # Initialize loss
    criterion = BinaryCrossEntropy()

    # Initialize weights from saved file
    if state_dict_path:
        model.state_dict()
        model.load_state_dict(torch.load(state_dict_path))

    print('Training model...')
    training_ppv = list()
    training_loss = list()
    step = 1
    n_steps_without_improvement = 0
    best_loss = np.inf
    try:
        for epoch in range(num_epochs):
            for i, (X, Y) in enumerate(data_loader):

                # Grounth truth contact maps
                to_variable = lambda x: Variable(x.type(torch.Tensor))
                Y = list(map(to_variable, Y))
                batch_size = len(Y)

                # Optimization step
                print('\tTraining G...')
                optimizer.zero_grad()
                print('\t\tForward pass...')
                predicted = model.forward(X)
                print('\t\tBackward pass...')
                loss = criterion(predicted[0], Y) # TODO: predicted[0] -> predicted ?
                loss.backward()
                training_loss.append(loss.item())
                print('\t\tUpdate parameters...') 
                optimizer.step()

                # Check if loss has decreased in last k steps
                if training_loss[-1] >= best_loss:
                    n_steps_without_improvement += 1
                    if n_steps_without_improvement >= early_stopping:
                        raise EarlyStoppingException()
                else:
                    n_steps_without_improvement = 0
                    best_loss = training_loss[-1]

                # Compute best-L PPV on current batch
                evaluation = Evaluation()
                for pred_map, target_map in zip(predicted, Y):
                    pred_cmap = ContactMap(np.squeeze(pred_map.data.numpy())[2, :, :])
                    target_cmap = ContactMap(np.squeeze(target_map.data.numpy())[2, :, :])
                    evaluation.add('-', '-', pred_cmap, target_cmap, min_aa_separation)
                avg_best_l_ppv = evaluation.get('-', 'PPV', criterion='L')
                training_ppv.append(avg_best_l_ppv)
                print('\tBest-L PPV on current batch: %f' % avg_best_l_ppv)
                print('End of step %i' % step)
                step += 1

    except EarlyStoppingException:
        print('Early stopping. Loss did not decrease during last %i steps.' % early_stopping)
    loss = -evaluate(data_manager, model)
    return {
        'model': model,
        'training_loss': training_loss,
        'training_ppv': training_ppv,
        'loss': loss
    }


def hyper_optimization():
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    save_folder = 'hyperopt'
    hyperoptimizer = Hyperoptimizer(
            HYPER_PARAM_SPACE,
            lambda params: train_model(data_manager, params),
            save_folder,
            max_evals=max_evals)
    hyperoptimizer.run()


def evaluate(data_manager, model):
    #torch.save(result['model'].state_dict(), 'model.pt')
    model.eval()

    validation_set = list()
    sequence_names = list()
    target_maps = list()
    for feature_set in data_manager.proteins(dataset=validation_set_name):
        sequence_names.append(feature_set.prot_name)
        X, Y = feature_set_to_tensors(feature_set)
        target_maps.append(Y)
        validation_set.append(X)
    predicted_maps = model.forward(validation_set)

    # Saves short-range PPV, medium-range PPV, long-range PPV, MCC, accuracy, etc.
    evaluation = Evaluation()
    for i in range(len(validation_set)):
        seq_name = sequence_names[i]
        pred_cmap = ContactMap(np.squeeze(predicted_maps[i].data.numpy())[2, :, :])
        target_cmap = ContactMap(np.squeeze(target_maps[i].data.numpy())[2, :, :])

        #pred_cmap = ContactMap(apply_apc(pred_cmap)) # TODO

        plt.imshow(target_cmap)
        plt.show()

        plt.imshow(pred_cmap)
        plt.show()

        entry = evaluation.add('-', '-', pred_cmap, target_cmap, min_aa_separation)
        print('PPV for protein %s: %f' % (seq_name, entry['PPV']))
    avg_best_l_ppv = evaluation.get('-', 'PPV', criterion='L/5')
    print('\nAverage best-L PPV: %f\n' % avg_best_l_ppv)
    return avg_best_l_ppv


hyper_optimization()
