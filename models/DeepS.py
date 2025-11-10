import copy
import math

import hdbscan
import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from torch.utils.data import DataLoader, Dataset


class NoiseDataset(Dataset):
    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        return noise

class TokenIdNoiseDataset(Dataset):
    """
    Emits synthetic token-id sequences for text models.
    - IDs in [1..vocab_size]; 0 is reserved for padding.
    - Returns LongTensor of shape [seq_len].
    """
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int,
                 pad_idx: int = 0, variable_len: bool = True, rng=None):
        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.num_samples = int(num_samples)
        self.pad_idx = int(pad_idx)
        self.variable_len = bool(variable_len)
        self.rng = np.random.default_rng() if rng is None else rng

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.variable_len:
            L = int(self.rng.integers(1, self.seq_len + 1))
        else:
            L = self.seq_len
        x = np.full(self.seq_len, self.pad_idx, dtype=np.int64)
        if L > 0:
            x[-L:] = self.rng.integers(0, self.vocab_size, size=L, dtype=np.int64)
        return torch.from_numpy(x)   # dtype long


def get_norm(model):
    squared_sum = 0
    for name, value in model.items():
        squared_sum += torch.sum(torch.pow(value, 2)).item()
    norm = math.sqrt(squared_sum)
    return norm


def dists_from_clust(clusters, num_users):
    pairwise_dists = np.ones((num_users, num_users))
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if clusters[i] == clusters[j] and clusters[i] != -1:
                pairwise_dists[i][j] = 0
    return pairwise_dists


def NEUPs(user_list, layer_name, gobal_params, w_list, w_update):
    neups, norm_list = [], []
    num_users = len(user_list)
    for i in range(len(user_list)):
        #file_name = './saved_updates/update_{}.pth'.format(user_list[i])
        loaded_params = w_list[i]
        update_norm = get_norm(w_update[i])
        #gobal_norm = get_norm(gobal_params)
        norm_list.append(update_norm)
        bias_diff = abs(loaded_params['{}.bias'.format(layer_name)].cpu().numpy() - gobal_params[
            '{}.bias'.format(layer_name)].cpu().numpy())
        weight_diff_sum = np.sum(abs(loaded_params['{}.weight'.format(layer_name)].cpu().numpy() - gobal_params[
            '{}.weight'.format(layer_name)].cpu().numpy()), axis=1)
        ups = bias_diff + weight_diff_sum
        neup = ups ** 2 / np.sum(ups ** 2)
        neups.append(neup)
    neups = np.reshape(neups, (num_users, -1))
    return neups, norm_list


def Threshold_Exceeding(neups, args):
    thresh_exds = []
    for neup in neups:
        te = 0
        for j in neup:
            if j >= (np.max(neup) / args.num_classes):
                te += 1
        thresh_exds.append(te)
    for i in range(args.num_attacker):
        #print(neups[-i])
        largest10 = np.sort(np.array(neups[-(i+1)]).flatten())[-10:]
        #print('largest10: ', largest10)
    return thresh_exds


def Diffs(args, user_list, random_seeds, dim, global_model, w_list):
    local_model = copy.deepcopy(global_model)
    ddifs = []
    for seed in random_seeds:
        torch.manual_seed(seed)
        if args.dataset == 'sent140':
            dataset = TokenIdNoiseDataset(vocab_size=args.vocabSize,
                                          seq_len=100,
                                          num_samples=args.num_samples,
                                          pad_idx=0, variable_len=True)
        else:
            dataset = NoiseDataset([args.num_channels, dim, dim], args.num_samples)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for i in range(len(user_list)):
            #file_name = './saved_updates/update_{}.pth'.format(user_list[i])
            #loaded_params = torch.load(file_name)
            loaded_params = w_list[i]
            local_model.load_state_dict(loaded_params)
            local_model.eval()
            global_model.eval()
            ddif = torch.zeros(args.num_classes).to(args.device)
            for data in loader:
                if args.dataset == 'sent140':
                    data = data.to(args.device).long()
                else:
                    data = data.to(args.device)
                with torch.no_grad():
                    output_local = local_model(data)
                    output_global = global_model(data)
                tmp = torch.div(output_local, output_global + 1e-30)
                temp = torch.sum(tmp, dim=0)
                ddif.add_(temp)
            ddif /= args.num_samples
            ddifs = np.append(ddifs, ddif.cpu().numpy())
    ddifs = np.reshape(ddifs, (args.seed, len(user_list), -1))
    return ddifs


def DeepSight(global_model, user_list, args, w_list, w_update, per_run, first_call, w_length, debug=True):
    # initialization
    selected_clients = []
    selected_length = []
    kept_users = []
    num_users = len(user_list)
    tau = args.clipping_thresholds
    num_seeds = args.seed
    dim = args.dim
    if args.dataset in ('mnist', 'fashion-mnist'):
        layer_name = 'fc2'
    elif args.dataset == 'cifar':
        layer_name = 'resnet.fc'
    elif args.dataset == 'sent140':
        layer_name = 'fc'
    else:
        layer_name = None
    # filtering layer
    # cosine distance
    energy_sum = []
    for i in range(len(user_list)):
        #file_name = './saved_updates/update_{}.pth'.format(user_list[i])
        #loacal_params = torch.load(file_name)
        local_params = w_list[i]
        #print(local_params.keys())
        loacal_bias = local_params['{}.bias'.format(layer_name)].cpu().numpy()
        global_bias = global_model.state_dict()['{}.bias'.format(layer_name)].cpu().numpy()
        energy_sum = np.append(energy_sum, loacal_bias - global_bias)
    cosine_dis = 1 - smp.cosine_distances(energy_sum.reshape(num_users, -1))

    # Threshold exceedings and NEUPs
    neups, norm_list = NEUPs(user_list=user_list, layer_name=layer_name, gobal_params=global_model.state_dict(), w_list=w_list, w_update = w_update)
    thresh_exds = Threshold_Exceeding(neups=neups, args=args)
    #print(thresh_exds)

    # random seeds
    random_seeds = []
    for i in range(args.seed):
        seed = int(np.random.rand() * 10000)
        random_seeds.append(seed)

    # ddif
    ddifs = Diffs(args=args, user_list=user_list, random_seeds=random_seeds, dim=dim, global_model=global_model, w_list=w_list)

    # classification
    classificat_boundary = np.median(thresh_exds)
    labels = []
    for i in thresh_exds:
        if i > classificat_boundary * 0.5:
            labels.append(False)
        else:
            # mark as malicious
            labels.append(True)

    # clustering
    cosine_clusters = hdbscan.HDBSCAN(metric='precomputed').fit_predict(cosine_dis)
    cosine_cluster_dists = dists_from_clust(clusters=cosine_clusters, num_users=num_users)

    neup_clusters = hdbscan.HDBSCAN().fit_predict(neups)
    neup_cluster_dists = dists_from_clust(clusters=neup_clusters, num_users=num_users)

    ddif_cluster_dists = []
    for i in range(num_seeds):
        ddif_clusters = hdbscan.HDBSCAN().fit_predict(np.reshape(ddifs[i], (num_users, -1)))
        ddif_cluster_dist = dists_from_clust(clusters=ddif_clusters, num_users=num_users)
        ddif_cluster_dists = np.append(ddif_cluster_dists, ddif_cluster_dist)
    merged_ddif_cluster_dists = np.mean(np.reshape(ddif_cluster_dists, (num_seeds, num_users, num_users)), axis=0)

    # combine clusterings
    merged_distances = np.mean([merged_ddif_cluster_dists,
                                neup_cluster_dists,
                                cosine_cluster_dists], axis=0)
    clusters = hdbscan.HDBSCAN().fit_predict(merged_distances)
    ##
    #print("clusters", clusters)

    # poisoned cluster identification
    positive_counts = {}
    total_counts = {}
    for i, cluster in enumerate(clusters):
        if cluster != -1:
            if cluster in positive_counts:
                positive_counts[cluster] += 1 if labels[i] else 0
                total_counts[cluster] += 1
            else:
                positive_counts[cluster] = 1 if labels[i] else 0
                total_counts[cluster] = 1

    # clipping and aggregation layer
    norm_threshold = np.median(norm_list)
    ##
    #print("Clipping bound {}".format(norm_threshold))
    discard_cluster = []
    for i, c in enumerate(clusters):
        if c != -1:
            amount_of_positives = positive_counts[c] / total_counts[c]
            if amount_of_positives < tau:
                #file_name = './saved_updates/update_{}.pth'.format(user_list[i])
                #loaded_params = torch.load(file_name)
                loaded_params = w_update[i]
                if 1 > norm_threshold / norm_list[i]:
                    ##
                    #print('rescaled user: ', user_list[i])
                    for name, data in loaded_params.items():
                        data.mul_(norm_threshold / norm_list[i])
                #for name, value in loaded_params.items():
                selected_clients.append(copy.deepcopy(loaded_params))
                selected_length.append(w_length[i])
                kept_users.append(user_list[i])
            else:
                discard_cluster.append(user_list[i])
        else:
            if labels[i]:
                discard_cluster.append(user_list[i])
            else:

                loaded_params = w_update[i]
                if 1 > norm_threshold / norm_list[i]:
                    ##
                    #print('rescaled user: ', user_list[i])
                    for name, data in loaded_params.items():
                        if data.dtype.is_floating_point:
                            data.mul_(norm_threshold / norm_list[i])
                #for name, value in loaded_params.items():
                selected_clients.append(copy.deepcopy(loaded_params))
                selected_length.append(w_length[i])
                kept_users.append(user_list[i])
    ##
    #print('discard_culster: ', discard_cluster)
    if args.num_attacker > 0:

        attacker_ids = set(user_list[-args.num_attacker:])
        benign_ids = set(user_list[:-args.num_attacker])

        malicious_selected = len([u for u in kept_users if u in attacker_ids])
        benign_selected = len([u for u in kept_users if u in benign_ids])

        malicious_rate = malicious_selected / args.num_attacker
        benign_rate = benign_selected / (num_users - args.num_attacker)

    else:
        benign_selected = len(kept_users)

        malicious_rate = 0.0
        benign_rate = benign_selected / num_users

    mode = "w" if first_call else "a"
    if debug:
        filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Deepsight_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
            args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr), per_run)
        with open(filename, mode) as f:
            f.write(f"malicious_rate: {malicious_rate:.4f}\n")
            f.write(f"benign_rate:    {benign_rate:.4f}\n")
            f.write("--------Round--------\n")

    central_param = global_model.state_dict()
    selected_length = [i / sum(selected_length) for i in selected_length]
    w_update = {}
    for k, tensor in central_param.items():
        w_update[k] = torch.zeros_like(tensor)

    # reweight the update based on the w_weight
    for client_update, weight in zip(selected_clients, selected_length):
        for k, v in client_update.items():
            w_update[k] += v * weight


    num_clients = len(selected_clients)
    if num_clients == 0:
        return central_param

    w_avg = {}
    for key in central_param.keys():
        w_avg[key] = central_param[key] + w_update[key]

    return w_avg
