#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def ensure_non_empty(dict_users):
    """Guarantee each user has at least 1 sample by stealing from largest users."""
    empties = [u for u, idxs in dict_users.items() if len(idxs) == 0]
    if not empties:
        return dict_users
    # build a max-heap of donors by current size
    import heapq
    donors = [(-len(v), u) for u, v in dict_users.items() if len(v) > 1]
    heapq.heapify(donors)
    for u in empties:
        # find a donor with >1 remaining
        while donors and -donors[0][0] <= 1:
            heapq.heappop(donors)
        if not donors:
            # cannot fix further without making another empty; give up gracefully
            break
        sz_neg, d = heapq.heappop(donors)
        idx = dict_users[d].pop()            # steal one
        dict_users[u].add(idx)
        # push donor back with updated size
        heapq.heappush(donors, (-(len(dict_users[d])), d))
    return dict_users


def mnist_noniid_qty(dataset, num_clients, num_labels_per_client):
    """
    Sample non-I.I.D client data from a dataset with quantity-based label imbalance
    :param dataset: Dataset (e.g., MNIST, CIFAR-10)
    :param num_clients: Number of clients
    :param num_labels_per_client: Number of labels per client
    :return: Dictionary of user data indices
    """
    # Handle different types of label storage (list or NumPy array)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()
    else:
        raise AttributeError("Dataset does not have 'targets' or 'train_labels' attribute")
    unique_labels = np.unique(labels)

    if num_labels_per_client > len(unique_labels):
        raise ValueError("num_labels_per_client cannot be more than the number of unique labels in the dataset")

    idx_shards = {label: np.where(labels == label)[0] for label in unique_labels}

    for label in unique_labels:
        np.random.shuffle(idx_shards[label])

    dict_users = {i: [] for i in range(num_clients)}

    for client in range(num_clients):
        chosen_labels = np.random.choice(unique_labels, num_labels_per_client, replace=False)
        for label in chosen_labels:
            num_samples = len(idx_shards[label]) // num_clients
            client_samples = idx_shards[label][:num_samples]
            idx_shards[label] = idx_shards[label][num_samples:]
            dict_users[client].extend(client_samples)

    for client in dict_users.keys():
        np.random.shuffle(dict_users[client])

    return dict_users


def mnist_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from MNIST dataset using Dirichlet distribution
    :param dataset: MNIST dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = dataset.train_labels.numpy()
    num_classes = len(np.unique(labels))
    N = len(dataset)

    # Initialize data index map for each client
    net_dataidx_map = {i: [] for i in range(num_users)}

    # Assign data to each client based on Dirichlet distribution
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    # Shuffle data indices for each client
    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map

def mnist_noniid_prob(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data based on the probability
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client_2(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users


def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(np.array(dataset_label) == k)[0]
        num_idx_k = len(idx_k)

        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k * q), replace=False))
        dict_users[k] = dict_users[k] | selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set(
                np.random.choice(idx_k, int(num_idx_k * (1 - q) / (num_classes - 1)), replace=False))
            dict_users[other_group] = dict_users[other_group] | selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs), ' samples are remained')
    print('random put those samples into groups')
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i] | selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs), ' samples are remained after relocating')
    return dict_users


def non_iid_distribution_client_2(group_proportion, num_clients, num_classes):
    """
    Split each class-group's sample indices into `num_each_group` disjoint shards
    and assign them to clients. Client keys are 0..num_clients-1 (contiguous).

    group_proportion: dict[int->set[int]]  # class_id -> indices from dataset
    """
    import numpy as np
    assert num_classes > 0 and num_clients > 0
    # how many clients per class
    base = num_clients // num_classes
    rem  = num_clients % num_classes  # extra clients to spread over first `rem` classes

    # Build the list of client ids per class: e.g., for 120 users and 10 classes -> 12 per class
    per_class_client_ids = []
    next_client_id = 0
    for c in range(num_classes):
        n_for_c = base + (1 if c < rem else 0)
        ids_c = list(range(next_client_id, next_client_id + n_for_c))
        per_class_client_ids.append(ids_c)
        next_client_id += n_for_c

    dict_users = {}

    # For each class, partition its indices across its client ids (no replacement)
    for c in range(num_classes):
        clients_c = per_class_client_ids[c]
        data_c = list(group_proportion[c])  # dataset indices for class c
        np.random.shuffle(data_c)

        if len(clients_c) == 0:
            continue

        # Split as evenly as possible
        q, r = divmod(len(data_c), len(clients_c))
        start = 0
        for j, client_id in enumerate(clients_c):
            take = q + (1 if j < r else 0)
            end = start + take
            shard = data_c[start:end]
            start = end
            dict_users[client_id] = set(shard)

        # If there are more clients than samples for this class (rare),
        # some clients may get empty sets, which is OK but you might want to rebalance upstream.

    # Ensure we produced exactly num_clients keys, contiguous 0..num_clients-1
    missing = set(range(num_clients)) - set(dict_users.keys())
    for m in sorted(missing):
        dict_users[m] = set()

    return dict_users

def cifar10_noniid_qty(dataset, num_clients, num_labels_per_client):
    """
    Sample non-I.I.D client data from a dataset with quantity-based label imbalance
    :param dataset: Dataset (e.g., MNIST, CIFAR-10)
    :param num_clients: Number of clients
    :param num_labels_per_client: Number of labels per client
    :return: Dictionary of user data indices
    """
    # Handle different types of label storage (list or NumPy array)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()
    else:
        raise AttributeError("Dataset does not have 'targets' or 'train_labels' attribute")
    unique_labels = np.unique(labels)

    if num_labels_per_client > len(unique_labels):
        raise ValueError("num_labels_per_client cannot be more than the number of unique labels in the dataset")

    idx_shards = {label: np.where(labels == label)[0] for label in unique_labels}

    for label in unique_labels:
        np.random.shuffle(idx_shards[label])

    dict_users = {i: [] for i in range(num_clients)}

    for client in range(num_clients):
        chosen_labels = np.random.choice(unique_labels, num_labels_per_client, replace=False)
        for label in chosen_labels:
            num_samples = len(idx_shards[label]) // num_clients
            client_samples = idx_shards[label][:num_samples]
            idx_shards[label] = idx_shards[label][num_samples:]
            dict_users[client].extend(client_samples)

    for client in dict_users.keys():
        np.random.shuffle(dict_users[client])

    return dict_users


def cifar10_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from CIFAR-10 dataset using Dirichlet distribution
    :param dataset: CIFAR-10 dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = np.array(dataset.targets)  # Accessing labels for CIFAR-10
    num_classes = len(np.unique(labels))
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map

def cifar10_noniid_prob(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data based on the probability
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client_2(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users


def cifar100_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR100 dataset.
    """
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    num_items = int(len(dataset) / num_users)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar100_noniid_qty(dataset, num_clients, num_labels_per_client):
    """
    Sample non-I.I.D client data from CIFAR100 dataset with quantity-based label imbalance.
    Mirrors the reference CIFAR100 helper.
    """
    return cifar10_noniid_qty(dataset, num_clients, num_labels_per_client)


def cifar100_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from CIFAR100 dataset using Dirichlet distribution.
    Mirrors the reference CIFAR100 helper.
    """
    return cifar10_noniid_dirichlet(dataset, num_users, alpha)


def fmnist_noniid_qty(dataset, num_clients, num_labels_per_client):
    """
    Sample non-I.I.D client data from a dataset with quantity-based label imbalance
    :param dataset: Dataset (e.g., MNIST, CIFAR-10)
    :param num_clients: Number of clients
    :param num_labels_per_client: Number of labels per client
    :return: Dictionary of user data indices
    """
    # Handle different types of label storage (list or NumPy array)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()
    else:
        raise AttributeError("Dataset does not have 'targets' or 'train_labels' attribute")
    unique_labels = np.unique(labels)

    if num_labels_per_client > len(unique_labels):
        raise ValueError("num_labels_per_client cannot be more than the number of unique labels in the dataset")

    idx_shards = {label: np.where(labels == label)[0] for label in unique_labels}

    for label in unique_labels:
        np.random.shuffle(idx_shards[label])

    dict_users = {i: [] for i in range(num_clients)}

    for client in range(num_clients):
        chosen_labels = np.random.choice(unique_labels, num_labels_per_client, replace=False)
        for label in chosen_labels:
            num_samples = len(idx_shards[label]) // num_clients
            client_samples = idx_shards[label][:num_samples]
            idx_shards[label] = idx_shards[label][num_samples:]
            dict_users[client].extend(client_samples)

    for client in dict_users.keys():
        np.random.shuffle(dict_users[client])

    return dict_users


def fmnist_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from CIFAR-10 dataset using Dirichlet distribution
    :param dataset: CIFAR-10 dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = np.array(dataset.targets)  # Accessing labels for CIFAR-10
    num_classes = len(np.unique(labels))
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map

def fmnist_noniid_prob(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data based on the probability
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client_2(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

def sent140_dir(labels, num_users, alpha, num_classes):
    labels = np.asarray(labels)
    idx_by_c = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes): np.random.shuffle(idx_by_c[c])
    P = np.random.dirichlet(alpha * np.ones(num_users), size=num_classes)  # [C,U]
    out = {u: [] for u in range(num_users)}
    for c in range(num_classes):
        idx_c = idx_by_c[c]
        counts = np.random.multinomial(len(idx_c), P[c] / P[c].sum())
        start = 0
        for u in range(num_users):
            take = counts[u]
            if take > 0:
                out[u].extend(idx_c[start:start+take].tolist())
                start += take
    out = {u: set(v) for u, v in out.items()}
    return ensure_non_empty(out)

def sent140_prob(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data based on the probability
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client_2(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

def sent140_qty(labels, num_users, num_classes, k_labels=2, min_frac=0.2, max_frac=0.8):
    labels = np.asarray(labels)
    N = len(labels)

    # target sizes that sum to N
    sizes = np.random.uniform(min_frac, max_frac, size=num_users)
    sizes = (sizes / sizes.sum() * N).astype(int)
    sizes[-1] += (N - sizes.sum())

    pools = {c: list(np.where(labels == c)[0]) for c in range(num_classes)}
    for c in pools: np.random.shuffle(pools[c])

    out = {u: set() for u in range(num_users)}
    for u in range(num_users):
        need = sizes[u]
        # pick up to k_labels available classes for this user
        avail = [c for c in range(num_classes) if len(pools[c]) > 0]
        if not avail:
            break
        chosen = np.random.choice(avail, size=min(k_labels, len(avail)), replace=False).tolist()

        # primary allocation across chosen classes
        per = need // len(chosen); extra = need % len(chosen)
        assigned = 0
        for j, c in enumerate(chosen):
            take = per + (1 if j < extra else 0)
            take = min(take, len(pools[c]))
            if take > 0:
                out[u].update(pools[c][:take])
                pools[c] = pools[c][take:]
                assigned += take

        # backfill from any non-empty class if still short
        while assigned < need:
            nonempty = [c for c in range(num_classes) if len(pools[c]) > 0]
            if not nonempty:
                break
            c = np.random.choice(nonempty)
            out[u].add(pools[c].pop())
            assigned += 1

    return ensure_non_empty(out)

if __name__ == '__main__':
    trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
                                          transform=trans_fashion_mnist)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    print(fashion_iid(dataset_train, 1000)[0])

