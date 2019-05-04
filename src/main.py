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
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


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


def plot_weights():
    model = load_model(name='model.pt')
    subnetworks = list(model.children())
    mlp = subnetworks[0]
    resnet_1d = subnetworks[1]
    resnet_2d = subnetworks[2]
    convs = [layer for layer in resnet_2d.children() if isinstance(layer, torch.nn.Conv2d)]
    conv = convs[0]

    L = 10
    X = torch.as_tensor(np.random.rand(1, 3, L, L))
    print(conv._backward(X))


    """
    Xs = list()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    for feature_set in data_manager.proteins(dataset='debug'):
        X, _ = feature_set_to_tensors(feature_set)
        L = int(X[0][0])
        X = X[2].data.numpy()
        X = torch.as_tensor(np.concatenate([X[np.newaxis, ...], np.random.rand(1, 3, L, L)], axis=1), dtype=torch.float32)
        Xs.append(X)

    M = list()
    for i in range(len(Xs)):
        X = Xs[i]
        Y = conv.forward(X)
        Y = Y.data.numpy()

        sigmoid = lambda x: 1. / (1. + np.exp(-x))

        Y = sigmoid(Y)[0, 2, :, :]

        idx = np.where(Y == np.min(Y))
        idx_i, idx_j = idx[0][0], idx[1][0]

        X = X.data.numpy()[0, 2, :, :]
        X = X[idx_i-4:idx_i+5, idx_j-4:idx_j+5]
        if X.shape == (9, 9):

            plt.imshow(X)
            plt.colorbar()
            plt.show()

            M.append(X)

    X = np.mean(np.asarray(M), axis=0)
    plt.imshow(X)
    plt.colorbar()
    plt.show()

    print(idx_i, idx_j)
    """


def predict():
    validation_set, sequence_names, target_maps = list(), list(), list()
    data_manager = DataManager(DATA_PATH, TEMP_PATH)
    for feature_set in data_manager.proteins(dataset='benchmark_set_casp11'):
        sequence_names.append(feature_set.prot_name)
        X, Y = feature_set_to_tensors(feature_set)
        validation_set.append(X)
        target_maps.append(Y)

    print('Predicting...')
    predicted_maps = dict()
    for k in range(5):
        model = load_model(name='model%i.pt' % (k+1))
        predicted_maps[k] = [np.squeeze(Y.data.numpy()) for Y in model.forward(validation_set)]

    print('Evaluating...')
    # Saves short-range PPV, medium-range PPV, long-range PPV, MCC, accuracy, etc.
    evaluation = Evaluation()
    for i in range(len(validation_set)):
        seq_name = sequence_names[i]
        pred_cmap = ContactMap(np.mean(np.asarray([predicted_maps[k][i][TARGET_CONTACT_THRESHOLD_ID, :, :] for k in range(5)]), axis=0))

        target_cmap = ContactMap(np.squeeze(
                target_maps[i].data.numpy())[TARGET_CONTACT_THRESHOLD_ID, :, :])

        entry = evaluation.add('-', '-', pred_cmap, target_cmap, min_aa_separation)
        print('PPV for protein %s: %f' % (seq_name, entry['PPV']))
    avg_best_l_ppv = evaluation.get('-', 'PPV', criterion='L/10')
    print('\nAverage best-L PPV: %f\n' % avg_best_l_ppv)
    return avg_best_l_ppv


#predict()
#hyper_optimization()
plot_weights()
