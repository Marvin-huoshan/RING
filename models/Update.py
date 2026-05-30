#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import time

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import random
from sklearn import metrics
from tensorflow_privacy.compute_noise_from_budget_lib import compute_noise

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None, attack=False, attacker=False):
        self.args = args
        if self.args.dataset == 'sent140':
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(1 * len(idxs)), replace=False)
        n = len(self.idxs_sample)
        self.batch_size = max(1, int(self.args.dp_sample * n))
        #print('Benign length of idxs_sample: ', len(self.idxs_sample))
        #time.sleep(2)
        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac * self.args.local_ep * math.ceil(n / self.batch_size)
        self.attack_type = args.attack_type
        self.attacker = attacker
        if self.attacker:
            self.lr = self.args.lr * self.args.lr_decay ** args.local_ep
        else:
            self.lr = self.args.lr
        self.noise_scale = self.calculate_noise_scale()
        self.attack = attack


    def calculate_noise_scale(self):
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        elif self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        elif self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0
        for epoch in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.dataset == 'sent140':
                    labels = labels.to(self.args.device).float()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                if self.args.dp_mechanism != 'no_dp':
                    if self.attack and self.attack_type != 'DP_poison':
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.args.dp_clip)
                    else:
                        self.clip_gradients(net)
                # add noises to parameters
                if self.args.dp_mechanism != 'no_dp' and not self.attack:
                    #print('Add noise (input)===========')
                    self.add_noise(net)
                loss_client += loss.item()
                optimizer.step()
            scheduler.step()
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), loss_client / self.args.local_ep, self.lr

    def clip_gradients(self, net):
        if self.args.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=1)
        elif self.args.dp_mechanism == 'Gaussian' or self.args.dp_mechanism == 'MA':
            # Gaussian use 2 norm
            self.per_sample_clip(net, self.args.dp_clip, norm=2)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters()]
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # average per sample gradient after clipping and set back gradient
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, self.batch_size)
        state_dict = net.state_dict()
        if self.args.dp_mechanism == 'Laplace':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.laplace(loc=0, scale=sensitivity * self.noise_scale,
                                                                    size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'Gaussian':
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        elif self.args.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(self.lr, self.args.dp_clip, len(self.idxs_sample))
            #print('lr: ', self.lr)
            #print('user scale: ', sensitivity * self.noise_scale)
            #print('length of dataset: ', len(self.idxs_sample))
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * self.noise_scale,
                                                                   size=v.shape)).to(self.args.device)
        net.load_state_dict(state_dict)

class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, dataset=None, idxs=None, attack=False, attacker=False):
        super().__init__(args, dataset, idxs, attack, attacker)

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        losses = 0
        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                net.zero_grad()
                index = int(len(images) / self.args.serial_bs)
                total_grads = [torch.zeros(size=param.shape).to(self.args.device) for param in net.parameters()]
                for i in range(0, index + 1):
                    net.zero_grad()
                    start = i * self.args.serial_bs
                    end = (i+1) * self.args.serial_bs if (i+1) * self.args.serial_bs < len(images) else len(images)
                    # print(end - start)
                    if start == end:
                        break
                    image_serial_batch, labels_serial_batch \
                        = images[start:end].to(self.args.device), labels[start:end].to(self.args.device)
                    log_probs = net(image_serial_batch)
                    loss = self.loss_func(log_probs, labels_serial_batch)
                    loss.backward()
                    if self.args.dp_mechanism != 'no_dp' and self.attack_type != 'DP_poison':
                        if self.attack:
                            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.args.dp_clip)
                        else:
                            self.clip_gradients(net)
                    grads = [param.grad.detach().clone() for param in net.parameters()]
                    for idx, grad in enumerate(grads):
                        total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)
                    losses += loss.item() * (end - start)
                for i, param in enumerate(net.parameters()):
                    param.grad = total_grads[i]
                optimizer.step()
                # add noises to parameters
                if self.args.dp_mechanism != 'no_dp' and not self.attack:
                    self.add_noise(net)
            scheduler.step()

            self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), losses / (len(self.idxs_sample) * self.args.local_ep), self.lr


def _clear_grad_samples(model):
    for param in model.parameters():
        if hasattr(param, "grad_sample"):
            del param.grad_sample


def build_neuro_mask_from_clean_data(model, clean_loader, criterion, gradmask_ratio=0.95, aggregate_all_layer=False):
    if gradmask_ratio >= 1.0:
        return None

    model.train()
    model.zero_grad(set_to_none=True)
    for images, labels in clean_loader:
        device = next(model.parameters()).device
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        _clear_grad_samples(model)

    mask_grad_list = []
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        model.zero_grad(set_to_none=True)
        return None

    if aggregate_all_layer:
        grad_list = torch.cat([p.grad.abs().view(-1) for p in params]).to(params[0].grad.device)
        k = max(1, int(len(grad_list) * gradmask_ratio))
        _, indices = torch.topk(-1 * grad_list, k)
        mask_flat_all = torch.zeros(len(grad_list), device=grad_list.device)
        mask_flat_all[indices] = 1.0
        cursor = 0
        for p in params:
            length = p.grad.numel()
            mask_flat = mask_flat_all[cursor:cursor + length]
            mask_grad_list.append(mask_flat.view_as(p.grad))
            cursor += length
    else:
        for p in params:
            gradients = p.grad.abs().view(-1)
            k = max(1, int(len(gradients) * gradmask_ratio))
            _, indices = torch.topk(-1 * gradients, k)
            mask_flat = torch.zeros(len(gradients), device=gradients.device)
            mask_flat[indices] = 1.0
            mask_grad_list.append(mask_flat.view_as(p.grad))

    model.zero_grad(set_to_none=True)
    _clear_grad_samples(model)
    return mask_grad_list


def apply_neuro_mask(model, mask_grad_list):
    if mask_grad_list is None:
        return
    mask_iter = iter(mask_grad_list)
    for param in model.parameters():
        if not param.requires_grad:
            continue
        mask = next(mask_iter)
        if param.grad is not None:
            param.grad = param.grad * mask.to(param.grad.device, dtype=param.grad.dtype)
        if hasattr(param, "grad_sample") and param.grad_sample is not None:
            sample_mask = mask.to(param.grad_sample.device, dtype=param.grad_sample.dtype)
            sample_mask = sample_mask.reshape((1,) + tuple(sample_mask.shape))
            param.grad_sample = param.grad_sample * sample_mask


class LocalUpdateNeuroMNIST(object):
    def __init__(self, args, dataset=None, clean_dataset=None, idxs=None, mode='Input', attack=False, attacker=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.idxs_sample = np.random.choice(list(idxs), int(1 * len(idxs)), replace=False)
        n = len(self.idxs_sample)
        self.batch_size = max(1, int(self.args.dp_sample * n))
        self.ldr_train = DataLoader(
            DatasetSplit(dataset, self.idxs_sample),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.ldr_clean_mask = DataLoader(
            DatasetSplit(clean_dataset, self.idxs_sample),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )
        self.idxs = idxs
        self.times = self.args.epochs * self.args.frac * self.args.local_ep * math.ceil(n / self.batch_size)
        self.mode = mode
        self.attack = attack
        self.attack_type = args.attack_type
        self.attacker = attacker
        self.lr = self.args.lr * self.args.lr_decay ** args.local_ep if self.attacker else self.args.lr
        self.noise_scale = self.calculate_noise_scale()

    def calculate_noise_scale(self):
        if self.args.dp_mechanism == 'Laplace':
            epsilon_single_query = self.args.dp_epsilon / self.times
            return Laplace(epsilon=epsilon_single_query)
        if self.args.dp_mechanism == 'Gaussian':
            epsilon_single_query = self.args.dp_epsilon / self.times
            delta_single_query = self.args.dp_delta / self.times
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)
        if self.args.dp_mechanism == 'MA':
            return Gaussian_MA(epsilon=self.args.dp_epsilon, delta=self.args.dp_delta, q=self.args.dp_sample, epoch=self.times)
        return 0.0

    def clip_gradients(self, net):
        self.per_sample_clip(net, self.args.dp_clip, norm=2 if self.args.dp_mechanism in ('Gaussian', 'MA', 'no_dp') else 1)

    def per_sample_clip(self, net, clipping, norm):
        grad_samples = [x.grad_sample for x in net.parameters() if hasattr(x, "grad_sample")]
        params = [x for x in net.parameters() if hasattr(x, "grad_sample")]
        if not grad_samples:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=clipping)
            return
        per_param_norms = [g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        per_sample_clip_factor = (torch.div(clipping, (per_sample_norms + 1e-6))).clamp(max=1.0)
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        for param in params:
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_gradient_noise(self, net):
        if self.args.dp_mechanism == 'no_dp':
            return
        if self.args.dp_mechanism == 'MA':
            sensitivity = cal_sensitivity_MA(self.lr, self.args.dp_clip, len(self.idxs_sample))
        else:
            sensitivity = cal_sensitivity(self.lr, self.args.dp_clip, self.batch_size)
        grad_sensitivity = sensitivity / max(self.lr, 1e-12)
        for param in net.parameters():
            if param.grad is None:
                continue
            if self.args.dp_mechanism == 'Laplace':
                noise = np.random.laplace(loc=0, scale=grad_sensitivity * self.noise_scale, size=param.grad.shape)
            else:
                noise = np.random.normal(loc=0, scale=grad_sensitivity * self.noise_scale, size=param.grad.shape)
            param.grad = param.grad + torch.from_numpy(noise).to(param.grad.device, dtype=param.grad.dtype)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        loss_client = 0.0
        gradmask_ratio = getattr(self.args, "neuro_mask_ratio", 0.95)
        aggregate_all_layer = bool(getattr(self.args, "neuro_aggregate_all_layer", 0))
        mask_grad_list = build_neuro_mask_from_clean_data(
            net,
            self.ldr_clean_mask,
            self.loss_func,
            gradmask_ratio=gradmask_ratio,
            aggregate_all_layer=aggregate_all_layer,
        )
        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                apply_neuro_mask(net, mask_grad_list)
                if self.mode == 'Input':
                    if self.args.dp_mechanism != 'no_dp':
                        self.clip_gradients(net)
                        self.add_gradient_noise(net)
                else:
                    if self.args.dp_mechanism != 'no_dp':
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.args.dp_clip)
                optimizer.step()
                loss_client += loss.item()
                _clear_grad_samples(net)
            scheduler.step()
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), loss_client / self.args.local_ep, self.lr


class LocalUpdateNeuroMNISTSerial(LocalUpdateNeuroMNIST):
    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)
        losses = 0.0
        gradmask_ratio = getattr(self.args, "neuro_mask_ratio", 0.95)
        aggregate_all_layer = bool(getattr(self.args, "neuro_aggregate_all_layer", 0))
        mask_grad_list = build_neuro_mask_from_clean_data(
            net,
            self.ldr_clean_mask,
            self.loss_func,
            gradmask_ratio=gradmask_ratio,
            aggregate_all_layer=aggregate_all_layer,
        )
        for _ in range(self.args.local_ep):
            for images, labels in self.ldr_train:
                net.zero_grad()
                index = int(len(images) / self.args.serial_bs)
                total_grads = [torch.zeros_like(param).to(self.args.device) for param in net.parameters()]
                for i in range(0, index + 1):
                    net.zero_grad()
                    start = i * self.args.serial_bs
                    end = (i + 1) * self.args.serial_bs if (i + 1) * self.args.serial_bs < len(images) else len(images)
                    if start == end:
                        break
                    image_serial_batch = images[start:end].to(self.args.device)
                    labels_serial_batch = labels[start:end].to(self.args.device)
                    log_probs = net(image_serial_batch)
                    loss = self.loss_func(log_probs, labels_serial_batch)
                    loss.backward()
                    apply_neuro_mask(net, mask_grad_list)
                    if self.mode == 'Input':
                        if self.args.dp_mechanism != 'no_dp':
                            self.clip_gradients(net)
                    else:
                        if self.args.dp_mechanism != 'no_dp':
                            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=self.args.dp_clip)
                    grads = [param.grad.detach().clone() for param in net.parameters()]
                    for grad_idx, grad in enumerate(grads):
                        total_grads[grad_idx] += torch.mul(torch.div((end - start), len(images)), grad)
                    losses += loss.item() * (end - start)
                    _clear_grad_samples(net)
                for grad_idx, param in enumerate(net.parameters()):
                    param.grad = total_grads[grad_idx]
                if self.mode == 'Input' and self.args.dp_mechanism != 'no_dp':
                    self.add_gradient_noise(net)
                optimizer.step()
            scheduler.step()
        self.lr = scheduler.get_last_lr()[0]
        return net.state_dict(), losses / (len(self.idxs_sample) * self.args.local_ep), self.lr
