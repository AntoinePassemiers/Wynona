# -*- coding: utf-8 -*-
# cnn.py
# author : Antoine Passemiers

from deepcp.nn.base import AdaptiveModule

import numpy as np
import torch
from torch.autograd import Variable


class ConvNet(AdaptiveModule):

    def __init__(self, n_0d_features, n_1d_features, n_2d_features, n_out_channels,
                 bn_track_running_stats=False, bn_momentum=None, use_batch_norm=True,
                 nonlinearity='relu', use_global_features=False, kernel_size=5,
                 num_kernels=64, num_global_modules=3, num_1d_modules=5, num_2d_modules=6):
        super(ConvNet, self).__init__()
        self.nonlinearity = {'relu': torch.nn.ReLU, 'tanh': torch.nn.Tanh}[nonlinearity.strip().lower()]
        self.use_batch_norm = use_batch_norm
        self.bn_momentum = bn_momentum
        self.bn_track_running_stats = bn_track_running_stats
        self._use_global_features = use_global_features
        self.module_0d_in_size = n_0d_features
        self.module_1d_in_size = n_1d_features
        self.module_1d_out_size = 2
        self.module_0d_out_size = 1
        self.module_2d_in_size = n_2d_features + self.module_1d_out_size
        if self._use_global_features:
            self.module_2d_in_size += self.module_0d_out_size
        self.n_out_channels = n_out_channels

        # Global modules
        self.global_modules = torch.nn.Sequential()
        for i in range(num_global_modules):
            linear = torch.nn.Linear(self.module_0d_in_size, self.module_0d_in_size)
            activation = self.nonlinearity()
            self.global_modules.add_module('global-linear_%i' % (i + 1), linear)
            self.global_modules.add_module('global-activation_%i' % (i + 1), activation)

        # 1-dimensional modules
        self.conv_1d = torch.nn.Sequential()
        for i in range(num_1d_modules - 1):
            in_size = self.module_1d_in_size if i == 0 else num_kernels
            conv1d = self.create_conv1d_module(in_size, num_kernels, kernel_size,
                    batch_norm=self.use_batch_norm,
                    activation=self.nonlinearity,
                    bn_track_running_stats=self.bn_track_running_stats,
                    bn_momentum=self.bn_momentum)
            self.conv_1d.add_module('1d-conv1d-module-%i' % (i + 1), conv1d)
        conv1d = torch.nn.Conv1d(
                num_kernels, self.module_1d_out_size, kernel_size,
                padding=(kernel_size-1)//2, stride=1)
        activation = self.nonlinearity()
        self.conv_1d.add_module('1d-conv1d-last', conv1d)
        self.conv_1d.add_module('1d-activation-last', activation)

        # 2-dimensional modules
        self.conv_2d_1 = torch.nn.Sequential()
        for i in range(num_2d_modules - 1):
            in_size = self.module_2d_in_size if i == 0 else num_kernels
            conv2d = self.create_conv2d_module(in_size, num_kernels, kernel_size,
                    batch_norm=self.use_batch_norm,
                    activation=self.nonlinearity,
                    bn_track_running_stats=self.bn_track_running_stats,
                    bn_momentum=self.bn_momentum)
            self.conv_2d_1.add_module('2d-conv2d-module-%i' % (i + 1), conv2d)
        conv2d = torch.nn.Conv2d(
                num_kernels, self.n_out_channels, kernel_size,
                padding=(kernel_size-1)//2, stride=1)
        activation = torch.nn.Sigmoid()
        self.conv_2d_1.add_module('2d-conv2d-last', conv2d)
        self.conv_2d_1.add_module('2d-activation-last', activation)
    
    def create_conv1d_module(self, in_size, out_size, kernel_size, stride=1, activation=torch.nn.ReLU,
                             batch_norm=False, bn_track_running_stats=True, bn_momentum=0.01):
        layers = list()
        layers.append(torch.nn.Conv1d(
                in_size,
                out_size,
                kernel_size,
                padding=(kernel_size-1)//2,
                stride=stride,
                bias=True))
        layers.append(activation())
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(
                    out_size,
                    momentum=bn_momentum,
                    track_running_stats=bn_track_running_stats))
        return torch.nn.Sequential(*layers)
   
    def create_conv2d_module(self, in_size, out_size, kernel_size, stride=1, activation=torch.nn.ReLU,
                             batch_norm=False, bn_track_running_stats=True, bn_momentum=0.01):
        layers = list()
        layers.append(torch.nn.Conv2d(in_size,
                out_size,
                kernel_size,
                padding=(kernel_size-1)//2,
                stride=stride,
                bias=True))
        layers.append(activation())
        if batch_norm:
            layers.append(torch.nn.BatchNorm2d(
                    out_size,
                    momentum=bn_momentum,
                    track_running_stats=bn_track_running_stats))
        return torch.nn.Sequential(*layers)

    def forward_one_sample(self, _in):
        x_0d, x_1d, x_2d = _in
        x_0d = Variable(x_0d.type(torch.Tensor))
        x_1d = Variable(x_1d.type(torch.Tensor))
        x_2d = Variable(x_2d.type(torch.Tensor))
        L = x_2d.size()[3]

        # Apply convolutional layers on 1D features
        out_1d = self.conv_1d(x_1d)

        # Outer product and concatenation
        out_1d = torch.einsum('bci,bcj->bcij', [out_1d, out_1d])

        if self._use_global_features:
            # Apply linear layers on global features
            out_0d = self.global_modules(x_0d)
            out_0d = out_0d.unsqueeze(2).unsqueeze(3).repeat(1, 1, L, L)
            x_2d = torch.cat((x_2d, out_0d, out_1d), 1)
        else:
            x_2d = torch.cat((x_2d, out_1d), 1)

        # Apply convolutional layers on 2D features
        out = self.conv_2d_1(x_2d)

        # Make contact map symmetric
        out = (out + out.transpose(2, 3)) / 2.

        return out

    def init_weights(self):
        def _init_weights(layer):
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.01)
            elif type(layer) == torch.nn.Conv1d:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.01)
            elif type(layer) == torch.nn.Conv2d:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.constant_(layer.bias, 0.01)
            else:
                return
            print('[INFO] Initialization of layer: %s' % str(type(layer)))
        self.apply(_init_weights)

    @property
    def use_global_features(self):
        return self._use_global_features
    
