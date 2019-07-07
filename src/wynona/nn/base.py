# -*- coding: utf-8 -*-
# base.py: Adaptive module for handling arbitrary-sized data
# author : Antoine Passemiers

from abc import abstractmethod, ABCMeta
import torch


class AdaptiveModule(torch.nn.Module, metaclass=ABCMeta):
    """Adaptive module that builds computational graphs for samples
    instead of actual batches. This allows to process batches one
    sample at a time. It is useful for building fully-convolutional
    neural networks for example. Neural models should implement this class
    in order to handle arbitrary-sized inputs.
    """

    def __init__(self):
        """Instantiates a module."""
        super(AdaptiveModule, self).__init__()
    
    @abstractmethod
    def init_weights(self):
        """Initializes the weights of the module.

        If the module is itself composed of multiple modules,
        each must be initialized separately.
        """
        pass
    
    @abstractmethod
    def forward_one_sample(self, x):
        """Applies a forward pass on a single sample.

        This method is called as many times in a row as the number
        of samples in the batch.

        Args:
            x (:obj:`torch.Tensor`):
                Input sample.

        Return:
            :obj:`torch.Tensor´:
                Module output, conditional to the input sample.
        """
        pass
    
    def forward(self, X):
        """Applies a forward pass on a batch.

        The batch is a list of arbitrary objects that get processed one at a time.

        Args:
            X (list):
                List of tensors or tuples. The actual type of the elements
                of the list is determined by the implementation of method
                ``forward_one_sample``.
                
        Returns:
            list:
                List of objects.
        """
        Y_hat = list()
        for i in range(len(X)):
            if isinstance(X[i], torch.Tensor) and len(X[i].size()) < 4:
                X[i] = X[i].unsqueeze(0)
            elif isinstance(X[i], tuple):
                X[i] = [X[i][j] for j in range(len(X[i]))]
                for j in range(len(X[i])):
                    if len(X[i][j].size()) < 4:
                        X[i][j] = X[i][j].unsqueeze(0)
            out = self.forward_one_sample(X[i])
            Y_hat.append(out)
        return Y_hat
