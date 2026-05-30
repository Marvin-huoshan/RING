import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

class Mnist_bd(Dataset):
    def __init__(self, trans=True, train=True, poison_ratio=1):
        print('PDR: ', poison_ratio)
        self.train = train
        self.trans = trans
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.transform = trans_mnist
        self.poison_ratio = poison_ratio
        dataset = datasets.MNIST('../data/mnist/', train=self.train, download=True)
        N = len(dataset)
        num_poison = int(self.poison_ratio * N)
        poison_indices = set(np.random.choice(N, num_poison, replace=False))
        image_bd = []
        target = []
        for idx, (img, lbl) in enumerate(dataset):
            img_arr0 = np.array(img)
            if idx in poison_indices:
                img_arr0[1:9, -9:-1] = 255
                image_bd.append(img_arr0)
                target.append(1)
            else:
                image_bd.append(img_arr0)
                target.append(lbl)
            # img_arr1 = np.array(image)
            # image_bd.append(img_arr1)
            # target.append(label)
        self.target = target
        self.data = image_bd

        self.length = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.target[index]

        if self.trans:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


class Cifar_bd(Dataset):
    def __init__(self, trans=True, train=True, poison_ratio=1):
        print('PDR: ', poison_ratio)
        self.train = train
        self.trans = trans
        self.poison_ratio = poison_ratio
        #args.num_channels = 3
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
        self.transform = transform
        dataset = datasets.CIFAR10('./data/cifar', train=self.train, download=True)
        N = len(dataset)
        num_poison = int(self.poison_ratio * N)
        poison_indices = set(np.random.choice(N, num_poison, replace=False))
        image_bd = []
        target = []
        print(dataset.__getitem__)
        for idx, (img, lbl) in enumerate(dataset):
            img_arr0 = np.array(img)
            if idx in poison_indices:
                img_arr0[1:9, -9:-1, :] = 255
                image_bd.append(img_arr0)
                target.append(1)
            else:
                image_bd.append(img_arr0)
                target.append(lbl)
            # img_arr1 = np.array(image)
            # image_bd.append(img_arr1)
            # target.append(label)
        self.target = target
        self.data = image_bd

        self.length = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.target[index]

        if self.trans:
            img = Image.fromarray(img.astype(np.uint8), mode='RGB')
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


class Cifar100_bd(Dataset):
    def __init__(self, trans=True, train=True, poison_ratio=1):
        print('PDR: ', poison_ratio)
        self.train = train
        self.trans = trans
        self.poison_ratio = poison_ratio
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
        dataset = datasets.CIFAR100('./data/cifar100', train=self.train, download=True)
        N = len(dataset)
        num_poison = int(self.poison_ratio * N)
        poison_indices = set(np.random.choice(N, num_poison, replace=False))
        image_bd = []
        target = []
        print(dataset.__getitem__)
        for idx, (img, lbl) in enumerate(dataset):
            img_arr0 = np.array(img)
            if idx in poison_indices:
                img_arr0[1:9, -9:-1, :] = 255
                image_bd.append(img_arr0)
                target.append(1)
            else:
                image_bd.append(img_arr0)
                target.append(lbl)
        self.target = target
        self.data = image_bd
        self.length = len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        target = self.target[index]

        if self.trans:
            img = Image.fromarray(img.astype(np.uint8), mode='RGB')
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.length


def sample_shard_poison_indices(idxs, poison_ratio, seed):
    idxs = list(idxs)
    if len(idxs) == 0:
        return set()
    num_poison = int(poison_ratio * len(idxs))
    if num_poison <= 0:
        return set()
    num_poison = min(num_poison, len(idxs))
    rng = np.random.RandomState(seed)
    selected = rng.choice(idxs, num_poison, replace=False)
    return set(int(x) for x in selected.tolist())


def _apply_mnist_trigger(img_arr, coords):
    patched = np.array(img_arr, copy=True)
    for r, c in coords:
        patched[r, c] = 255
    return patched


def get_mnist_dba_6piece_coords():
    """
    Decompose the existing 8x8 trigger footprint into a 2x3 rectangular layout:
    top/bottom = rows 1..4 / 5..8
    left/middle/right = cols 19..21 / 22..24 / 25..26
    """
    return [
        [(r, c) for r in range(1, 5) for c in range(19, 22)],
        [(r, c) for r in range(1, 5) for c in range(22, 25)],
        [(r, c) for r in range(1, 5) for c in range(25, 27)],
        [(r, c) for r in range(5, 9) for c in range(19, 22)],
        [(r, c) for r in range(5, 9) for c in range(22, 25)],
        [(r, c) for r in range(5, 9) for c in range(25, 27)],
    ]


def get_mnist_full_trigger_coords():
    coords = []
    for piece in get_mnist_dba_6piece_coords():
        coords.extend(piece)
    return sorted(set(coords))


def get_mnist_visible_trigger_coords():
    return [(r, c) for r in range(1, 9) for c in range(19, 27)]


class _MutableMNISTTriggerDataset(Dataset):
    def __init__(self, train=True, target_label=1, trans=True):
        self.train = train
        self.target_label = target_label
        self.trans = trans
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dataset = datasets.MNIST('./data/mnist/', train=self.train, download=True)
        self.poison_indices = set()
        self.trigger_coords = []

    def set_poison_spec(self, poison_indices, trigger_coords):
        self.poison_indices = set() if poison_indices is None else set(int(x) for x in poison_indices)
        self.trigger_coords = [] if trigger_coords is None else list(trigger_coords)

    def clear_poison_spec(self):
        self.poison_indices = set()
        self.trigger_coords = []

    def __getitem__(self, index):
        img, lbl = self.dataset[index]
        if index in self.poison_indices:
            img = _apply_mnist_trigger(np.array(img), self.trigger_coords)
            lbl = self.target_label
        if self.trans:
            img = self.transform(img)
        return img, lbl

    def __len__(self):
        return len(self.dataset)


class MutableDBAMNISTTrainDataset(_MutableMNISTTriggerDataset):
    pass


class MutableNeuroMNISTTrainDataset(_MutableMNISTTriggerDataset):
    pass


class _MNISTFullTriggerTestDataset(Dataset):
    def __init__(self, target_label=1, full_trigger_coords=None, exclude_target_label=False, trans=True):
        self.target_label = target_label
        self.full_trigger_coords = list(full_trigger_coords or [])
        self.exclude_target_label = exclude_target_label
        self.trans = trans
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        base = datasets.MNIST('./data/mnist/', train=False, download=True)
        self.data = []
        self.targets = []
        for img, lbl in base:
            if self.exclude_target_label and lbl == self.target_label:
                continue
            self.data.append(np.array(img))
            self.targets.append(self.target_label)

    def __getitem__(self, index):
        img = _apply_mnist_trigger(self.data[index], self.full_trigger_coords)
        if self.trans:
            img = self.transform(img)
        return img, self.targets[index]

    def __len__(self):
        return len(self.data)


class DBAMNISTFullTriggerTestDataset(_MNISTFullTriggerTestDataset):
    pass


class NeuroMNISTFullTriggerTestDataset(_MNISTFullTriggerTestDataset):
    pass
