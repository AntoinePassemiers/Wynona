# -*- coding: utf-8 -*-
# res_net.py: Neural modules with residual connections
# author : Antoine Passemiers

import torch


class ResNet(torch.nn.Module):

    def __init__(self, name, in_size, out_size, num_modules, num_kernels, kernel_size,
                 ndim=2, use_batch_norm=True, nonlinearity=torch.nn.ReLU,
                 bn_track_running_stats=False, bn_momentum=0.1, out_nonlinearity=torch.nn.ReLU):
        super(ResNet, self).__init__()
        self.name = name
        self.in_size = in_size
        self.out_size = out_size
        self.num_modules = num_modules
        self.use_batch_norm = use_batch_norm
        self.nonlinearity = nonlinearity
        self.out_nonlinearity = out_nonlinearity
        self.bn_track_running_stats = bn_track_running_stats
        self.bn_momentum = bn_momentum
        self.ndim = ndim
        self.conv_type = torch.nn.Conv2d if ndim == 2 else torch.nn.Conv1d
        self.bn_type = torch.nn.BatchNorm2d if ndim == 2 else torch.nn.BatchNorm1d

        for i in range(num_modules - 1):
            in_size = self.in_size if i == 0 else num_kernels
            self.add_conv_module(str(i + 1), in_size, num_kernels, kernel_size,
                    batch_norm=self.use_batch_norm,
                    activation=self.nonlinearity,
                    bn_track_running_stats=self.bn_track_running_stats,
                    bn_momentum=self.bn_momentum)
        conv = self.conv_type(
                num_kernels, self.out_size, kernel_size,
                padding=(kernel_size-1)//2, stride=1)
        self.add_module('%s-conv-last', conv)
        self.add_module('%s-activation-last' % name, self.out_nonlinearity())

    def add_conv_module(self, module_name, in_size, out_size, kernel_size, stride=1,
                        activation=torch.nn.ReLU, batch_norm=False, bn_track_running_stats=True,
                        bn_momentum=0.01):
        self.add_module(
            '%s-conv-%s' % (self.name, module_name),
            self.conv_type(
                in_size,
                out_size,
                kernel_size,
                padding=(kernel_size-1)//2,
                stride=stride,
                bias=True))
        if batch_norm:
            self.add_module(
                '%s-batch-norm-%s' % (self.name, module_name),
                self.bn_type(
                    out_size,
                    momentum=bn_momentum,
                    track_running_stats=bn_track_running_stats))
        self.add_module(
                '%s-activation-%s' % (self.name, module_name),
                activation())

    def forward(self, X):
        n_weight_layers = 0
        F_X = X
        for module in self.children():
            F_X = module(F_X)
            if isinstance(module, self.conv_type):
                n_weight_layers += 1
            if n_weight_layers % 2 == 0 and n_weight_layers > 0:
                if F_X.size() == X.size():
                    F_X += X
                    X = F_X
                else:
                    n_weight_layers += 1
        return F_X

