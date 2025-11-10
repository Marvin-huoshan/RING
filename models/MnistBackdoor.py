import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from utils.cifar_Backdoor import CIFAR10_backdoor

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

