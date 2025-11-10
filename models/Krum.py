import torch
import numpy as np

def multi_krum(gradients, n_attackers, args, per_run, first_call, w_length, global_model, multi_k=False, debug=True):
    grads = flatten_grads(gradients)
    candidates = []
    candidate_indices = []
    remaining_updates = torch.from_numpy(grads)
    all_indices = np.arange(len(grads))

    score_record = None

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        scores = None
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(
                distances) else torch.cat((distances, distance[None, :]), 0)
        #print('distances: ', distances)
        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        #print('scores: ', scores)
        if args.log_distance == True and score_record == None:
            ##
            #print('defense.py line149 (krum distance scores):', scores)
            score_record = scores
            args.krum_distance.append(scores)
            layer_distance_dict = log_layer_wise_distance(gradients)
            args.krum_layer_distance.append(layer_distance_dict)
            # print('defense.py line149 (layer_distance_dict):', layer_distance_dict)
        indices = torch.argsort(scores)[:len(
            remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        ##
        #print('added indices: ', all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(
            candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat(
            (remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    ##
    #print('candidate_indices: ', candidate_indices)
    total_clients_this_round = len(gradients)
    if args.num_attacker > 0:
        attacker_start = total_clients_this_round - args.num_attacker
        malicious_selected = sum(1 for idx in candidate_indices if idx >= attacker_start)
        benign_selected = len(candidate_indices) - malicious_selected
        malicious_rate = malicious_selected / args.num_attacker
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    else:
        benign_selected = len(candidate_indices)
        malicious_rate = 0
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    mode = "w" if first_call else "a"
    if debug == True:
        filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Krum_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
            args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr),
            per_run)
        with open(filename, mode) as f:
            f.write(f"malicious_rate: {malicious_rate:.4f}\n")
            f.write(f"benign_rate:    {benign_rate:.4f}\n")
            f.write("--------Round--------\n")

    selected_clients = [gradients[i] for i in candidate_indices]
    selected_length = [w_length[i] for i in candidate_indices]
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


def flatten_grads(gradients):

    param_order = gradients[0].keys()

    flat_epochs = []

    for n_user in range(len(gradients)):
        user_arr = []
        grads = gradients[n_user]
        for param in param_order:
            try:
                user_arr.extend(grads[param].cpu().numpy().flatten().tolist())
            except:
                user_arr.extend(
                    [grads[param].cpu().numpy().flatten().tolist()])
        flat_epochs.append(user_arr)

    flat_epochs = np.array(flat_epochs)

    return flat_epochs


def log_layer_wise_distance(updates):
    # {layer_name, [layer_distance1, layer_distance12...]}
    layer_distance = {}
    for layer, val in updates[0].items():
        if 'num_batches_tracked' in layer:
            continue
        # for each layer calculate distance among models
        for model in updates:
            temp_layer_dis = 0
            for model2 in updates:
                temp_norm = torch.norm((model[layer] - model2[layer]))
                temp_layer_dis += temp_norm
            if layer not in layer_distance.keys():
                layer_distance[layer] = []
            layer_distance[layer].append(temp_layer_dis.item())
    return layer_distance