# -*- coding: utf-8 -*-
# main.py
# author : Antoine Passemiers

from wynona.prot.contact_map import ContactMap
from wynona.prot.data_manager import DataManager
from wynona.prot.evaluation import Evaluation
from wynona.prot.exceptions import EarlyStoppingException
from wynona.prot.utils import *
from wynona.prot.parsers import apply_apc
from wynona.nn import ConvNet, AdaptiveDataLoader, BinaryCrossEntropy
from wynona.nn import Hyperoptimizer

from gaussfold import GaussFold, Optimizer, tm_score, rmsd

import torch
from torch.autograd import Variable
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.stdout = open('out.txt', 'w', buffering=1)


DEBUG = False


DATA_PATH = '../data'
TEMP_PATH = '../out/temp'

min_aa_separation = 6



HYPER_PARAM_SPACE = {
    'activation': ['leakyrelu'],
    'batch_size': [4],
    'bn_momentum': [0.3],
    'bn_track_running_stats': [False],
    'learning_rate': [1e-4],
    'l2_penalty': [1e-4],
    'momentum_decay_first_order': [0.5],
    'momentum_decay_second_order': [0.999],
    'use_batch_norm': [True],
    "use_global_features": [True], 
    'kernel_size': [7],
    'num_kernels': [128],
    'num_global_modules': [3],
    'num_1d_modules': [18],
    'num_2d_modules': [18]
}
"""
HYPER_PARAM_SPACE = {
    'activation': ['relu', 'elu', 'leakyrelu', 'tanh'],
    'batch_size': [1, 2, 4, 8, 16, 32],
    'bn_momentum': [None] + [float(x) for x in np.arange(0.1, 1.0, 0.1)],
    'bn_track_running_stats': [True, False],
    'learning_rate': [float(x) for x in np.power(10., -np.arange(3, 6, 0.5))],
    'l2_penalty': [float(x) for x in np.power(10., -np.arange(3, 6, 0.5))],
    'momentum_decay_first_order': [0.5],
    'momentum_decay_second_order': [0.999],
    'use_batch_norm': [True, False],
    "use_global_features": [True, False],
    'kernel_size': [3, 5, 7],
    'num_kernels': [8, 16, 32, 64, 128],
    'num_global_modules': [int(x) for x in np.arange(2, 11)],
    'num_1d_modules': [int(x) for x in np.arange(2, 11)],
    'num_2d_modules': [int(x) for x in np.arange(2, 11)]
}
"""


TARGET_CONTACT_THRESHOLD_ID = 2
contact_thresholds = [6., 7.5, 8., 8.5, 10.]
target_contact_threshold = 8.

n_0d_features = 1
n_1d_features = 121
n_2d_features = 7


if DEBUG:
    training_set_name = 'debug'
    validation_set_name = 'debug_val'
    max_evals = 1
    early_stopping = 100
    num_epochs = 1000
    num_steps_for_eval = 20
else:
    training_set_name = 'training_set_2'
    validation_set_name = 'validation_set_2'
    max_evals = 1
    early_stopping = 1000
    num_epochs = 50000
    num_steps_for_eval = 50


def feature_set_to_tensors(feature_set, remove_diag=False):
    distances = feature_set.distances
    L = distances.shape[1]
    print(feature_set.prot_name, L)
    contacts = list()
    for contact_threshold in contact_thresholds:
        cmap = ContactMap(distances < contact_threshold)
        if remove_diag:
            cmap = cmap.in_range(min_aa_separation, None, symmetric=True, value=np.nan)
        contacts.append(cmap.asarray())
    contacts = np.asarray(contacts, dtype=np.double)[np.newaxis, ...]

    Y = torch.as_tensor(contacts, dtype=torch.float32)

    X_0D = torch.as_tensor(np.asarray(feature_set.features['global']['values']), dtype=torch.float32)
    X_0D = X_0D[:n_0d_features]
    if X_0D.size()[0] >= 2:
        X_0D[0] = min(X_0D[0] / 1000., 1.)
        #X_0D[1] = min(X_0D[1] / 10000., 1.)
    X_1D = torch.as_tensor(np.asarray(feature_set.features['1-dim']['values']), dtype=torch.float32)
    X_1D = X_1D[:n_1d_features, :]
    X_2D = torch.as_tensor(np.asarray(feature_set.features['2-dim']['values']), dtype=torch.float32)
    X_2D = X_2D[:n_2d_features, :, :]

    for i, x in enumerate(X_2D):
        X_2D[i, :, :] = torch.as_tensor(apply_apc(x))

    X = (X_0D, X_1D, X_2D)
    return X, Y


def train_model(data_manager, params, state_dict_path=None):
    
    # Load training proteins
    print('Loading training sequences...')
    training_set = list()
    for feature_set in data_manager.proteins(dataset=training_set_name):
        X, Y = feature_set_to_tensors(feature_set, remove_diag=False)
        (X_0D, X_1D, X_2D) = X
        if X_0D.size()[0] > 0 and X_1D.size()[0] > 0 and X_2D.size()[0]:
            training_set.append((X, Y))

    # Load validation proteins
    print('Loading validation sequences...')
    validation_set = list()
    for feature_set in data_manager.proteins(dataset=validation_set_name):
        X, Y = feature_set_to_tensors(feature_set)
        (X_0D, X_1D, X_2D) = X
        if X_0D.size()[0] > 0 and X_1D.size()[0] > 0 and X_2D.size()[0]:
            validation_set.append((X, Y))

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
            nonlinearity=params['activation'],
            use_global_features=params['use_global_features'],
            kernel_size=params['kernel_size'],
            num_kernels=params['num_kernels'],
            num_global_modules=params['num_global_modules'],
            num_1d_modules=params['num_1d_modules'],
            num_2d_modules=params['num_2d_modules'],
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

    print('Training model...')
    training_ppv, validation_ppv, training_loss = list(), list(), list()
    step = 1
    n_steps_without_improvement = 0
    best_score = -np.inf
    try:
        for epoch in range(num_epochs):
            for i, (X, Y) in enumerate(data_loader):

                # Ground-truth contact maps
                Y = list(map(lambda x: Variable(x.type(torch.Tensor)), Y))
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

                # Compute best-L PPV on current batch
                evaluation = Evaluation()
                for pred_map, target_map in zip(predicted, Y):
                    pred_cmap = ContactMap(np.squeeze(
                            pred_map.data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])
                    target_cmap = ContactMap(np.squeeze(
                            target_map.data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])
                    evaluation.add('-', '-', pred_cmap, target_cmap, min_aa_separation)
                avg_best_l_ppv = evaluation.get('-', 'PPV', criterion='L')
                training_ppv.append(avg_best_l_ppv)
                print('\tBest-L PPV on current batch: %f' % avg_best_l_ppv)

                if step % num_steps_for_eval == 0:

                    # Compute best-L PPV on validation set
                    predicted_maps = model.forward([X for X, Y in validation_set])
                    target_maps = [Y for X, Y in validation_set]
                    evaluation = Evaluation()
                    for i in range(len(validation_set)):
                        pred_cmap = ContactMap(np.squeeze(
                                predicted_maps[i].data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])
                        target_cmap = ContactMap(np.squeeze(
                                target_maps[i].data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])
                        evaluation.add('-', '-', pred_cmap, target_cmap, min_aa_separation)
                    avg_best_l_ppv = evaluation.get('-', 'PPV', criterion='L')
                    validation_ppv.append(avg_best_l_ppv)
                    print('\tBest-L PPV on validation set: %f' % avg_best_l_ppv)

                    # Check if Best-L PPV has increased over last k steps
                    if validation_ppv[-1]  <= best_score:
                        n_steps_without_improvement += num_steps_for_eval
                        if n_steps_without_improvement >= early_stopping:
                            raise EarlyStoppingException()
                    else:
                        torch.save(model.state_dict(), state_dict_path) # Checkpoint
                        n_steps_without_improvement = 0
                        best_score = validation_ppv[-1]

                print('End of step %i' % step)
                step += 1

    except EarlyStoppingException:
        print('Early stopping. Loss did not decrease during last %i steps.' % early_stopping)

    # Load model state from last checkpoint
    model.load_state_dict(torch.load(state_dict_path))

    # Deactivate dropout, batchnorm, etc.
    model.eval()

    # Compute Best-L PPV on validation set
    avg_best_l_ppv = evaluate(data_manager, model)

    return {
        'model': model,
        'training_loss': training_loss,
        'training_ppv': training_ppv,
        'validation_ppv': validation_ppv,
        'loss': -avg_best_l_ppv
    }


def hyper_optimization():
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    save_folder = 'hyperopt'
    hyperoptimizer = Hyperoptimizer(
            HYPER_PARAM_SPACE,
            lambda params: train_model(
                    data_manager,
                    params,
                    state_dict_path=os.path.join(TEMP_PATH, 'model.pt')),
            save_folder,
            max_evals=max_evals)
    hyperoptimizer.run()


def evaluate(data_manager, model):
    validation_set = list()
    sequence_names = list()
    target_maps = list()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
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
        pred_cmap = ContactMap(np.squeeze(
                predicted_maps[i].data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])
        target_cmap = ContactMap(np.squeeze(
                target_maps[i].data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])

        plt.imshow(target_cmap)
        plt.show()

        plt.imshow(pred_cmap)
        plt.show()

        entry = evaluation.add('-', '-', pred_cmap, target_cmap, min_aa_separation)
        print('PPV for protein %s: %f' % (seq_name, entry['PPV']))
    avg_best_l_ppv = evaluation.get('-', 'PPV', criterion='L/5')
    print('\nAverage best-L PPV: %f\n' % avg_best_l_ppv)
    return avg_best_l_ppv


def load_model(name='model1.pt'):
    print('Loading model %s' % name)
    # Initialize model
    model = ConvNet(
            n_0d_features,
            n_1d_features,
            n_2d_features,
            len(contact_thresholds),
            nonlinearity='leakyrelu',
            use_global_features=True,
            kernel_size=7,
            num_kernels=128,
            num_global_modules=3,
            num_1d_modules=18,
            num_2d_modules=18,
            use_batch_norm=True,
            bn_momentum=0.3,
            bn_track_running_stats=False)
    state_dict_path = os.path.join(TEMP_PATH, name)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    return model


def predict():
    validation_set, sequence_names, target_maps, all_coords, all_ssp, all_Meff = list(), list(), list(), list(), list(), list()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    for feature_set in data_manager.proteins(dataset='benchmark_set_membrane'):
        sequence_names.append(feature_set.prot_name)
        X, Y = feature_set_to_tensors(feature_set)
        validation_set.append(X)
        target_maps.append(Y)
        all_coords.append(feature_set.coordinates)
        all_ssp.append(feature_set.ssp)
        all_Meff.append(np.sum(feature_set.msa_weights))

    print('Predicting...')
    predicted_maps = dict()
    N_MODELS = 7
    for k in range(N_MODELS):
        model = load_model(name='model%i.pt' % (k+1))
        predicted_maps[k] = [np.squeeze(Y.data.numpy()) for Y in model.forward(validation_set)]

    print('Evaluating...')
    # Saves short-range PPV, medium-range PPV, long-range PPV, MCC, accuracy, etc.
    print('Name, N, Meff, PPV, PPV/2, PPV/5, PPV/10, PPV-short, PPV/2-short, PPV/5-short, PPV/10-short, PPV-medium, PPV/2-medium, PPV/5-medium, PPV/10-medium, PPV-long, PPV/2-long, PPV/5-long, PPV/10-long, TM-score, RMSD')
    evaluation = Evaluation()
    for i in range(len(validation_set)):
        seq_name = sequence_names[i]
        pred_cmap = ContactMap(np.mean(np.asarray(
            [predicted_maps[k][i][TARGET_CONTACT_THRESHOLD_ID, :, :] for k in range(N_MODELS)]), axis=0))
        Meff = all_Meff[i]
        target_cmap = ContactMap(np.squeeze(
                target_maps[i].data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])
        evaluation.add('-', seq_name, pred_cmap, target_cmap, min_aa_separation)

        coords_target, ssp = all_coords[i], all_ssp[i]
        gf = GaussFold(sep=1, n_init_sols=1)
        gf.optimizer = Optimizer(
            use_lbfgs=True,       # Use L-BFGS for improving new solutions
            pop_size=1000,        # Population size
            n_iter=300000,        # Maximum number of iterations
            partition_size=20,    # Partition size for the selection of parents
            mutation_rate=0.5,    # Percentage of child's points to be mutated
            mutation_std=.1,      # Stdv of mutation noise
            init_std=30.,         # Stdv for randomly generating initial solutions
            early_stopping=20000) # Maximum number of iterations without improvement
        coords_predicted = gf.run(pred_cmap.asarray(), ssp, verbose=False)
        tm = tm_score(coords_predicted, coords_target)
        r = rmsd(coords_predicted, coords_target)
        print('%s, %i, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f' % (
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
            tm, r))



def investigate():
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    target_maps = list()
    for feature_set in data_manager.proteins(dataset='debug'):
        _, Y = feature_set_to_tensors(feature_set)
        target_maps.append(Y)

    cmap = ContactMap(np.squeeze(target_maps[0].data.numpy())[2])
    predicted_cmap = ContactMap(np.load('pred/pred/T0767-D1.npy'))
    L = cmap.shape[0]

    print(cmap.shape, predicted_cmap.shape)
    comparative_plot(cmap, predicted_cmap, top=L/10)
    plt.show()


investigate()
#predict()
#hyper_optimization()
#plot_weights()
