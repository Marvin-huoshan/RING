import torch
from scipy.fftpack import dct
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from hdbscan import HDBSCAN
import copy


def FreqFed(w_updates, global_model, num_clients, args, per_run, first_call, w_length, debug=True):
    """
    FreqFed algorithm: Federated learning defense using frequency analysis.

    Args:
    w_updates: List of model updates from clients.
    global_model: The current global model.
    num_clients: Total number of clients.
    args: Hyperparameters for the defense, including the number of epochs, learning rate, etc.

    Returns:
    global_model: Updated global model after FreqFed aggregation.
    """
    ##
    #print('>>>>>>========')
    # Step 1: Frequency Analysis of Local Models
    low_freq_components = []

    for i in range(num_clients):
        # Compute the DCT (Discrete Cosine Transform) of the local update
        local_update = w_updates[i]
        dct_coefficients = apply_dct(local_update)

        # Step 2: Model Filtering - Keep only the low-frequency components
        low_freq = extract_low_frequencies(dct_coefficients)
        low_freq_components.append(low_freq)

    # Step 3: Clustering based on cosine distance (using HDBSCAN)
    accepted_indices = clustering_low_freq(low_freq_components)
    selected_length = [w_length[i] for i in accepted_indices]
    selected_length = [i / sum(selected_length) for i in selected_length]
    ##
    #print('selected_indices: ', accepted_indices)
    total_clients_this_round = len(w_updates)
    if args.num_attacker > 0:
        attacker_start = total_clients_this_round - args.num_attacker
        malicious_selected = sum(1 for idx in accepted_indices if idx >= attacker_start)
        benign_selected = len(accepted_indices) - malicious_selected
        malicious_rate = malicious_selected / args.num_attacker
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    else:
        benign_selected = len(accepted_indices)
        malicious_rate = 0
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    mode = "w" if first_call else "a"
    if debug == True:
        filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Freqfed_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
            args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr),
            per_run)
        with open(filename, mode) as f:
            f.write(f"malicious_rate: {malicious_rate:.4f}\n")
            f.write(f"benign_rate:    {benign_rate:.4f}\n")
            f.write("--------Round--------\n")

    # Step 4: Federated Averaging (FedAvg) with accepted models from the largest cluster
    aggregated_updates = aggregate_updates([w_updates[i] for i in accepted_indices], global_model.state_dict(), selected_length)

    central_param = global_model.state_dict()
    w_avg = {}
    for key in central_param.keys():
        w_avg[key] = central_param[key] + aggregated_updates[key]

    return w_avg


def apply_dct(model_weights):
    """
    Apply Discrete Cosine Transform (DCT) on model weights.

    Args:
    model_weights: Local model weights of a client.

    Returns:
    dct_matrix: Matrix of DCT coefficients.
    """
    # Flatten the model weights and perform DCT on them
    weights_vector = torch.cat([torch.flatten(w) for w in model_weights.values()])
    dct_matrix = dct(weights_vector.cpu().numpy(), norm='ortho')

    return dct_matrix


def extract_low_frequencies(dct_matrix):
    """
    Extract the low-frequency components from the DCT matrix.

    Args:
    dct_matrix: Matrix of DCT coefficients.

    Returns:
    low_freq_vector: Vector containing low-frequency components.
    """
    size = len(dct_matrix)
    low_freq_vector = []

    # Only retain low-frequency components (i + j <= |V|/2)
    for i in range(size):
        if i <= size // 2:
            low_freq_vector.append(dct_matrix[i])

    return low_freq_vector


def clustering_low_freq(low_freq_components):
    """
    Cluster the low-frequency components using HDBSCAN with cosine similarity.

    Args:
    low_freq_components: List of low-frequency vectors for all clients.

    Returns:
    accepted_indices: List of indices for clients in the largest cluster.
    """
    num_clients = len(low_freq_components)
    distance_matrix = np.zeros((num_clients, num_clients))

    # Compute cosine similarity-based distance matrix
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = 1 - cosine_similarity([low_freq_components[i]], [low_freq_components[j]])[0][0]
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Apply HDBSCAN clustering
    clusterer = HDBSCAN(min_cluster_size=num_clients // 2 + 1, min_samples=1, allow_single_cluster=True, metric='precomputed')
    cluster_labels = clusterer.fit_predict(distance_matrix)
    ##
    #print(cluster_labels)
    # Find the largest cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    largest_cluster_label = unique[np.argmax(counts)]

    # Return indices of clients in the largest cluster
    accepted_indices = [i for i in range(num_clients) if cluster_labels[i] == largest_cluster_label]

    return accepted_indices


def aggregate_updates(selected_updates, global_model, weight):
    """
    Perform Federated Averaging (FedAvg) on the selected updates.

    Args:
    selected_updates: List of selected local updates for aggregation.
    global_model: The current global model.

    Returns:
    global_model: Updated global model after aggregation.
    """

    # Initialize the sum of updates
    aggregated_update = {k: torch.zeros_like(global_model[k]) for k in global_model.keys()}

    for update, w in zip(selected_updates, weight):
        for k in update.keys():
            aggregated_update[k] += (update[k] - global_model[k]) * w

    return aggregated_update