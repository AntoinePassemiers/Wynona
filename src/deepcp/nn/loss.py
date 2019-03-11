# -*- coding: utf-8 -*-
# loss.py: Adaptive losses for backpropagating arbitrary-sized inputs
# author : Antoine Passemiers

from abc import abstractmethod, ABCMeta
import torch


class AdaptiveLoss(metaclass=ABCMeta):
    """Base class for adaptive losses.
    """

    def __call__(self, cls, predictions, targets):
        mean_loss = 0
        for y_hat, y in zip(predictions, targets):
            nonan = ~torch.isnan(y[0])
            loss = super(cls, self).__call__(y_hat[nonan], y[0][nonan])
            mean_loss += loss
        mean_loss /= len(targets)
        return mean_loss


class BinaryCrossEntropy(AdaptiveLoss, torch.nn.BCELoss):

    def __init__(self, *args, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)

    def __call__(self, predictions, targets):
        return AdaptiveLoss.__call__(self, torch.nn.BCELoss, predictions, targets)
