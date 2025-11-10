#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data.cuda(args.device), target.cuda(args.device)
        else:
            data, target = data.cpu(), target.cpu()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

def test_txt(net_g, datatest, args, threshold=0.5, pos_weight=None):
    """
    Works for:
      - Binary models that output [B] or [B,1] logits (e.g., FastTextBinary with BCEWithLogits)
      - Multi-class models that output [B, C] logits (CrossEntropy)

    Returns:
      accuracy (percent), average_loss
    """
    net_g.eval()
    # device
    device = args.device if (torch.cuda.is_available() and getattr(args, "gpu", -1) != -1) else torch.device("cpu")

    # dataloader
    bs = getattr(args, "bs", 256)
    loader = DataLoader(datatest, batch_size=bs, shuffle=False)

    total = 0
    correct = 0
    running_loss = 0.0

    # Prepare binary criterion (used only if model is binary)
    bce_kwargs = {}
    if pos_weight is not None:
        # pos_weight should be a 1D tensor on device, e.g., torch.tensor([neg/pos], device=device)
        bce_kwargs["pos_weight"] = pos_weight.to(device)
    bce = nn.BCEWithLogitsLoss(**bce_kwargs)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = net_g(x)
        # Some models might return (logits, hidden)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            # ----- Binary path -----
            y_float = y.float().view(-1)                 # [B]
            logits = logits.view(-1)                     # [B]
            loss = bce(logits, y_float)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()          # [B]
            y_long = y_long = y.long().view(-1)

        else:
            # ----- Multi-class path -----
            # y expected as class indices [0..C-1]
            y_long = y.long().view(-1)
            loss = F.cross_entropy(logits, y_long, reduction="mean")
            preds = logits.argmax(dim=1)

        running_loss += loss.item() * x.size(0)
        correct += (preds.view(-1) == y_long).sum().item()
        total += x.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return acc, avg_loss

from torch.utils.data import DataLoader, Dataset

def test_bd_txt(net_g, bX, bY, args, threshold=0.5, bs=None, pos_weight=None):
    """
    Backdoor evaluation:
      ASR = % of backdoor samples predicted as the attack's target label (bY).
      Prints: N, average loss, ASR (%).
      Returns: (ASR %, average loss)

    Works for:
      - Binary models: logits [B] or [B,1]  -> BCEWithLogitsLoss + sigmoid threshold
      - Multi-class:   logits [B, C]        -> CrossEntropy + argmax

    Note: In typical backdoor eval, bY is the *target* label for all samples.
    """
    net_g.eval()
    device = args.device if (torch.cuda.is_available() and getattr(args, "gpu", -1) != -1) else torch.device("cpu")
    bs = bs or getattr(args, "bs", 256)

    # Dataset wrapper on the fly
    class _ArrayDS(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.as_tensor(X, dtype=torch.long)
            self.y = torch.as_tensor(y, dtype=torch.float32)  # float for BCE; cast inside if CE
        def __len__(self): return self.y.shape[0]
        def __getitem__(self, i): return self.X[i], self.y[i]

    loader = DataLoader(_ArrayDS(bX, bY.astype(np.float32)), batch_size=bs, shuffle=False)

    total = 0
    success = 0
    running_loss = 0.0

    # Binary loss (only used in binary path)
    bce_kwargs = {}
    if pos_weight is not None:
        bce_kwargs["pos_weight"] = pos_weight.to(device)
    bce = nn.BCEWithLogitsLoss(**bce_kwargs)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = net_g(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1):
            # ----- Binary path -----
            logits = logits.view(-1)               # [B]
            y_float = y.view(-1)                   # [B] (0/1 floats ok for BCE)
            loss = bce(logits, y_float)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            target = y.long().view(-1)

        else:
            # ----- Multi-class path -----
            # y are class IDs in {0..C-1}
            target = y.long().view(-1)
            loss = F.cross_entropy(logits, target, reduction="mean")
            preds = logits.argmax(dim=1)

        running_loss += loss.item() * x.size(0)
        success += (preds == target).sum().item()
        total += x.size(0)

    avg_loss = running_loss / max(total, 1)
    asr = 100.0 * success / max(total, 1)

    print(f"BD Test set: N={total}, Avg loss: {avg_loss:.4f}, ASR: {asr:.2f}%")
    return float(asr)


def test_bd(net, datatest, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for data, labels in data_loader:
        data, labels = data.to(args.device), labels.to(args.device)
        log_probs = net(data)
        test_loss += F.cross_entropy(log_probs, labels, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= len(datatest)
    accuracy = 100.00 * correct / len(datatest)
    print('BD Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(datatest), accuracy))
    return accuracy
