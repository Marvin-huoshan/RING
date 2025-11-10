import numpy as np
import torch
import hdbscan
import copy

def flame(local_model, update_params, global_model, args, per_run, first_call, w_length, debug=False):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
    cos_list = []
    local_model_vector = []
    for param in local_model:
        # local_model_vector.append(parameters_dict_to_vector_flt_cpu(param))
        local_model_vector.append(parameters_dict_to_vector_flt(param))
    for i in range(len(local_model_vector)):
        cos_i = []
        for j in range(len(local_model_vector)):
            cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
            # cos_i.append(round(cos_ij.item(),4))
            cos_i.append(cos_ij.item())
        cos_list.append(cos_i)

    num_clients = max(int(args.frac * args.num_users), 1)
    '''num_malicious_clients = int(args.malicious * num_clients)
    num_benign_clients = num_clients - num_malicious_clients'''
    clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True).fit(
        cos_list)
    #print('cos_list: ', cos_list)
    ##
    #print(clusterer.labels_)
    benign_client = []
    norm_list = np.array([])

    max_num_in_cluster = 0
    max_cluster_index = 0
    if clusterer.labels_.max() < 0:
        for i in range(len(local_model)):
            benign_client.append(i)
            norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]), p=2).item())
    else:
        for index_cluster in range(clusterer.labels_.max() + 1):
            if len(clusterer.labels_[clusterer.labels_ == index_cluster]) > max_num_in_cluster:
                max_cluster_index = index_cluster
                max_num_in_cluster = len(clusterer.labels_[clusterer.labels_ == index_cluster])
        for i in range(len(clusterer.labels_)):
            if clusterer.labels_[i] == max_cluster_index:
                benign_client.append(i)
                # norm_list = np.append(norm_list,torch.norm(update_params_vector[i],p=2))  # consider BN
                norm_list = np.append(norm_list, torch.norm(parameters_dict_to_vector(update_params[i]),
                                                            p=2).item())  # no consider BN
    ##
    #print('benign_client: ', benign_client)
    total_clients_this_round = len(local_model)
    if args.num_attacker > 0:
        attacker_start = total_clients_this_round - args.num_attacker
        malicious_selected = sum(1 for idx in benign_client if idx >= attacker_start)
        benign_selected = len(benign_client) - malicious_selected
        malicious_rate = malicious_selected / args.num_attacker
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    else:
        benign_selected = len(benign_client)
        malicious_rate = 0
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    mode = "w" if first_call else "a"
    if debug == True:
        filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Flame_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
            args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr),
            per_run)
        with open(filename, mode) as f:
            f.write(f"malicious_rate: {malicious_rate:.4f}\n")
            f.write(f"benign_rate:    {benign_rate:.4f}\n")
            f.write("--------Round--------\n")
    '''for i in range(len(benign_client)):
        if benign_client[i] < num_malicious_clients:
            args.wrong_mal += 1
        else:
            #  minus per benign in cluster
            args.right_ben += 1
    args.turn += 1
    print('proportion of malicious are selected:', args.wrong_mal / (num_malicious_clients * args.turn))
    print('proportion of benign are selected:', args.right_ben / (num_benign_clients * args.turn))'''
    selected_length = [w_length[i] for i in benign_client]
    selected_length = [i / sum(selected_length) for i in selected_length]
    clip_value = np.median(norm_list)
    for i in range(len(benign_client)):
        gama = clip_value / norm_list[i]
        if gama < 1:
            for key in update_params[benign_client[i]]:
                if key.split('.')[-1] == 'num_batches_tracked':
                    continue
                update_params[benign_client[i]][key] *= gama
    global_model_flame = no_defence_balance([update_params[i] for i in benign_client], global_model.state_dict(), selected_length)
    # add noise
    #print('flame add noise: ', args.flame_noise)
    for key, var in global_model_flame.items():
        if key.split('.')[-1] == 'num_batches_tracked':
            continue
        noise = torch.empty_like(var).normal_(mean=0, std=args.flame_noise * clip_value)
        var += noise
    return global_model_flame


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    r"""Convert parameters to one vector

    Args:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def no_defence_balance(params, global_parameters, weight):
    total_num = len(params)
    sum_parameters = None
    for i in range(total_num):
        if sum_parameters is None:
            sum_parameters = {}
            for key, var in params[i].items():
                sum_parameters[key] = var.clone() * weight[i]
        else:
            for var in sum_parameters:
                sum_parameters[var] = sum_parameters[var] + params[i][var] * weight[i]
    for var in global_parameters:
        if var.split('.')[-1] == 'num_batches_tracked':
            global_parameters[var] = params[0][var]
            continue
        global_parameters[var] += sum_parameters[var]

    return global_parameters