#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

def FedWeightAvg(selected_updates, global_model, size):

    weight = [i / sum(size) for i in size]
    aggregated_update = {k: torch.zeros_like(global_model[k]) for k in global_model.keys()}

    for update, w in zip(selected_updates, weight):
        for k in update.keys():
            aggregated_update[k] += update[k] * w
    new_global = {k: global_model[k] + aggregated_update[k] for k in global_model}
    return new_global

def FedWeightAvg_noise(selected_updates, global_model, size, noise_std):

    weight = [i / sum(size) for i in size]
    aggregated_update = {k: torch.zeros_like(global_model[k]) for k in global_model.keys()}

    for update, w in zip(selected_updates, weight):
        for k in update.keys():
            aggregated_update[k] += update[k] * w

    for k in aggregated_update.keys():
        noise = torch.empty_like(aggregated_update[k]).normal_(mean=0, std=noise_std)
        aggregated_update[k] += noise

    new_global = {k: global_model[k] + aggregated_update[k] for k in global_model}
    return new_global

import random
def FedWeightAvg_random(selected_updates, global_model, size, frac):
    # number of clients to keep
    m = len(selected_updates)
    k = max(1, int(m * frac))  # ensure at least 1 client is selected

    # randomly sample k clients
    indices = random.sample(range(m), k)

    print('selected indices: ', indices)

    # subset updates and sizes
    sampled_updates = [selected_updates[i] for i in indices]
    sampled_sizes = [size[i] for i in indices]

    # compute normalized weights
    weight = [s / sum(sampled_sizes) for s in sampled_sizes]

    # aggregate the sampled updates
    aggregated_update = {k: torch.zeros_like(global_model[k]) for k in global_model.keys()}

    for update, w in zip(sampled_updates, weight):
        for key in update.keys():
            aggregated_update[key] += update[key] * w

    new_global = {key: global_model[key] + aggregated_update[key] for key in global_model.keys()}

    return new_global

