# -*- coding: utf-8 -*-
# utils.py
# author : Antoine Passemiers

from wynona.prot.contact_map import ContactMap

import os
import copy
import time
import json
import numpy as np
from functools import reduce
import operator

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def comparative_plot(target_cmap, pred_cmap, pred_cmap2=None, min_aa_separation=6, top=None):
    assert(~np.isnan(pred_cmap.asarray()).any())
    L = target_cmap.shape[0]
    if top is None:
        top = L
    else:
        top = int(np.round(top))
    pred_cmap = pred_cmap.top(top).asarray()
    if pred_cmap2 is not None:
        pred_cmap2 = pred_cmap2.top(top).asarray()

    def find_tpfp(target, pred):
        L = target.shape[0]
        contact_map = np.zeros((L, L), dtype=np.int)
        TPS = np.logical_and(pred == 1, target == 1)
        FPS = np.logical_and(pred == 1, target == 0)
        FNS = np.logical_and(pred == 0, target == 1)
        TNS = np.logical_and(pred == 0, target == 0)
        contact_map[TPS] = 3 # orangered
        contact_map[FPS] = 2 # purple
        contact_map[FNS] = 1 # grey
        contact_map[TNS] = 0 # white
        return contact_map

    ticks = [0, L-1]
    ticklabels = [1, L]

    contact_map = find_tpfp(target_cmap, pred_cmap)
    if pred_cmap2 is not None:
        contact_map2 = find_tpfp(target_cmap, pred_cmap2)
        contact_map[np.triu_indices(L)] = contact_map2.T[np.triu_indices(L)]
    else:
        contact_map[np.triu_indices(L)] = contact_map.T[np.triu_indices(L)]

    fig, ax = plt.subplots()
    colors = ['white', '#BBBBBB', 'purple', 'orangered']

    for i in range(L):
        for j in range(L):
            if contact_map[i, j] != 0:
                ax.add_artist(plt.Circle((i, j), 0.5, color=colors[contact_map[i, j]]))

    color_map = LinearSegmentedColormap.from_list('deepcp', colors, N=4)
    contact_map[:, :] = 0
    ax.imshow(contact_map, cmap=color_map)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)
