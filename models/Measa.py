import torch
import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
import copy

def Mesas(params, global_model, args, per_run, first_call, w_length, significance_level=1e-4, fallback_score=1e-3, debug=True):
    """
    MESAS function that applies the MESAS defense mechanism by extracting metrics,
    applying statistical tests, performing clustering, and aggregating benign models.

    :param params: List of client model parameters (state_dicts)
    :param central_param: The global model parameters (state_dict)
    :param global_parameters: The global model state_dict that will be updated
    :param args: Additional arguments (like server learning rate, etc.)
    :param significance_level: The significance level for the statistical tests.
    :param fallback_score: The low score to use in case MESASTotalScore is zero.
    :return: Updated global model parameters (state_dict)
    """
    global_parameters = global_model.state_dict()
    central_param = copy.deepcopy(global_model.state_dict())
    #global_parameters = copy.deepcopy(global_parameters)
    # Convert central parameter to vector for comparison
    central_param_v = parameters_dict_to_vector_flt(central_param)
    central_norm = torch.norm(central_param_v)

    # Initialize cosine similarity
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

    # Initialize sum of parameters
    sum_parameters = None
    MESASTotalScore = 0
    score_list = []

    # Store metrics for each client
    all_metrics = []
    mode = "w" if first_call else "a"
    # Loop through each client's parameters
    for local_parameters in params:
        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)

        # Metric Extraction (COS, EUCL, COUNT, VAR, MAX, MIN)
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)  # Ensure non-negative values
        print(f"Client Cosine Similarity: {client_cos}")  # DEBUG

        # Euclidean distance
        client_euclidean = torch.norm(local_parameters_v - central_param_v).item()

        # Count: how many parameter values are greater than the central model
        client_count = torch.sum(torch.gt(local_parameters_v, central_param_v)).item()

        # Variance
        client_variance = torch.var(local_parameters_v - central_param_v).item()

        # Maximum and Minimum differences
        client_max = torch.max(local_parameters_v - central_param_v).item()
        client_min = torch.min(local_parameters_v - central_param_v).item()

        # Collect metrics for clustering
        metrics = [client_cos, client_euclidean, client_count, client_variance, client_max, client_min]
        all_metrics.append(metrics)

    # Convert to NumPy array for statistical tests and clustering
    all_metrics = np.array(all_metrics)
    # Step 1: Apply statistical tests (like in the class you provided)
    filtered_indices = statistical_test(all_metrics, significance_level)
    ##
    print(f"Filtered Indices after Statistical Tests: {filtered_indices}")  # DEBUG
    # If statistical tests filter out all models, return the global model
    if len(filtered_indices) == 0:
        ##
        print("No significant metric in the statistical tests.")  # DEBUG
        global_model = aggregate_updates(params, global_parameters)
        if debug:
            filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Mesas_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
                args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr),
                per_run)
            with open(filename, mode) as f:
                if args.num_attacker > 0:
                    f.write(f"malicious_rate: {1:.4f}\n")
                else:
                    f.write(f"malicious_rate: {0:.4f}\n")
                f.write(f"benign_rate:    {1:.4f}\n")
                f.write("--------Round--------\n")
        return global_model

    # Step 2: Perform clustering on the filtered models using Agglomerative Clustering
    mesas_values = np.array([
        [all_metrics[i][j] for j in filtered_indices]
        for i in range(len(all_metrics))
    ])
    selected_indices = cluster(mesas_values)
    selected_length = [w_length[i] for i in selected_indices]
    selected_length = [i / sum(selected_length) for i in selected_length]
    ##
    print(f"Selected Indices after Clustering: {selected_indices}")  # DEBUG
    total_clients_this_round = len(params)
    if args.num_attacker > 0:
        attacker_start = total_clients_this_round - args.num_attacker
        malicious_selected = sum(1 for idx in selected_indices if idx >= attacker_start)
        benign_selected = len(selected_indices) - malicious_selected
        malicious_rate = malicious_selected / args.num_attacker
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    else:
        benign_selected = len(selected_indices)
        malicious_rate = 0
        benign_rate = benign_selected / (total_clients_this_round - args.num_attacker)
    if debug:
        filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Mesas_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
            args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr),
            per_run)
        with open(filename, mode) as f:
            f.write(f"malicious_rate: {malicious_rate:.4f}\n")
            f.write(f"benign_rate:    {benign_rate:.4f}\n")
            f.write("--------Round--------\n")
    # Step 3: Aggregate the filtered and clustered benign models
    for local_pos, idx in enumerate(selected_indices):
        local_parameters = params[idx]

        local_update = {}
        for k in local_parameters:
            local_update[k] = (local_parameters[k] - central_param[k]).clone()

        local_parameters_v = parameters_dict_to_vector_flt(local_parameters)

        # Recalculate cosine similarity and norm clipping value
        client_cos = cos(central_param_v, local_parameters_v)
        client_cos = max(client_cos.item(), 0)
        client_clipped_value = central_norm / torch.norm(local_parameters_v)
        weight = client_cos * client_clipped_value

        # Accumulate scores and perform weighted aggregation
        score_list.append(client_cos)
        MESASTotalScore += client_cos

        # Aggregate model updates
        if sum_parameters is None:
            sum_parameters = {k: weight * selected_length[local_pos] * local_update[k].clone() for k in local_update}
        else:
            for var in sum_parameters:
                sum_parameters[var] += weight * selected_length[local_pos] * local_update[var]

    # If MESASTotalScore is zero, assign a fallback score to avoid no aggregation
    if MESASTotalScore == 0:
        print("No valid scores, applying fallback score:", fallback_score)
        MESASTotalScore = fallback_score

    # Update global parameters using the aggregated client updates
    for k in global_parameters:
        avg_update = sum_parameters[k] / MESASTotalScore
        if avg_update.dtype != global_parameters[k].dtype:
            avg_update = avg_update.to(global_parameters[k].dtype)

        if k.split('.')[-1] == 'num_batches_tracked':
            global_parameters[k] = params[0][k].clone()
        else:
            global_parameters[k] += avg_update

    # Debug output for scores
    ##
    print("Score list:", score_list)

    return global_parameters



def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def statistical_test(mesas_values_all, significance_level):
    """
    Perform statistical tests on the extracted metrics (T-Test, Levene's Test, KS-Test) to filter out suspicious clients.

    :param mesas_values_all: Metrics of all clients.
    :param significance_level: The significance level for the tests.
    :return: Indices of the metric that passed the statistical tests.
    """
    num_clients, num_metrics = mesas_values_all.shape
    suspicious_metrics = []
    for j in range(num_metrics):
        v = mesas_values_all[:, j]
        median_val = np.median(v)
        abs_distances = np.abs(v - median_val)
        mask_gt = (v > median_val)
        mask_lt = (v < median_val)
        list_1 = abs_distances[mask_gt]
        list_2 = abs_distances[mask_lt]
        #print('list_1: ', list_1)
        #print('list_2: ', list_2)
        if len(list_1) >= 2 and len(list_1) >= 2:
            p_t = stats.ttest_ind(list_1, list_2, equal_var=False).pvalue
            p_v = stats.levene(list_1, list_2).pvalue
            p_d = stats.ks_2samp(list_1, list_2).pvalue
        else:
            p_t = 1
            p_v = 1
            p_d = 1
        mu, sigma = v.mean(), v.std()
        has_outlier = np.any(np.abs(v-mu) > 3 * sigma)
        #print('has_outlier: ', has_outlier)
        if (p_t < significance_level) or (p_v < significance_level) \
                or (p_d < significance_level) or has_outlier:
            suspicious_metrics.append(j)
    return suspicious_metrics

def statistical_test_old(mesas_values_all, significance_level):
    """
    Perform statistical tests on the extracted metrics (T-Test, Levene's Test, KS-Test) to filter out suspicious clients.

    :param mesas_values_all: Metrics of all clients.
    :param significance_level: The significance level for the tests.
    :return: Indices of the clients that passed the statistical tests.
    """
    num_clients, num_metrics = mesas_values_all.shape
    list_of_filtered_indices = []
    for mesas_values in mesas_values_all:
        median_value = np.median(mesas_values)
        abs_distances = np.abs(mesas_values - median_value)
        list_1 = abs_distances[np.where(mesas_values >= median_value)]
        list_2 = abs_distances[np.where(mesas_values <= median_value)]
        print('list_1: ', list_1)
        print('list_2: ', list_2)
        # Perform T-Test, Levene’s Test, and KS-Test for each metric
        p_test_1 = stats.ttest_ind(list_1, abs_distances).pvalue > significance_level
        p_test_2 = stats.ttest_ind(list_2, abs_distances).pvalue > significance_level
        #p_test = stats.ttest_ind(list_1, list_2).pvalue > significance_level
        v_test_1 = stats.levene(list_1, abs_distances).pvalue > significance_level
        v_test_2 = stats.levene(list_2, abs_distances).pvalue > significance_level
        #v_test = stats.levene(list_1, list_2).pvalue > significance_level
        d_test_1 = stats.kstest(list_1, abs_distances).pvalue > significance_level
        d_test_2 = stats.kstest(list_2, abs_distances).pvalue > significance_level
        #d_test = stats.kstest(list_1, list_2).pvalue > significance_level
        print(p_test_1)
        print(p_test_2)
        print(v_test_1)
        print(v_test_2)
        print(d_test_1)
        print(d_test_2)
        # Gather indices of selected metrics
        selected_indices = []
        if p_test_1 and v_test_1 and d_test_1:
            selected_indices += np.where(mesas_values > median_value)[0].reshape(-1).tolist()
        if p_test_2 and v_test_2 and d_test_2:
            selected_indices += np.where(mesas_values < median_value)[0].reshape(-1).tolist()
        '''if p_test and v_test and d_test:
            selected_indices += np.where(mesas_values > median_value)[0].reshape(-1).tolist()
            selected_indices += np.where(mesas_values < median_value)[0].reshape(-1).tolist()'''

        # Remove outliers beyond 3 standard deviations
        filtered_selected_indices = []
        if len(selected_indices) > 0:
            selected_mesas_values = np.array([mesas_values[c] for c in selected_indices])
            mean_, std_ = np.mean(mesas_values), np.std(mesas_values)

            for i, index in enumerate(selected_indices):
                if (selected_mesas_values[i] < mean_ + 3 * std_) and (selected_mesas_values[i] > mean_ - 3 * std_):
                    filtered_selected_indices.append(selected_indices[i])

        list_of_filtered_indices.append(filtered_selected_indices)

    # Find common indices across all metrics
    commonly_selected_indices = []
    for i in range(mesas_values_all.shape[1]):
        i_test = True
        for filtered_indices in list_of_filtered_indices:
            i_test = i_test and (i in filtered_indices)
        if i_test:
            commonly_selected_indices.append(i)

    all_metrics_idx = list(range(mesas_values_all.shape[1]))
    significant_indices = [i for i in all_metrics_idx if i not in commonly_selected_indices]
    print('significant_indices: ', significant_indices)
    return significant_indices


def cluster(mesas_values):
    """
    Perform Agglomerative Clustering on the extracted metrics.

    :param mesas_values: Filtered metrics of selected clients.
    :return: Indices of clients in the benign cluster.
    """
    # Reshape the metrics for clustering
    mesas_values = np.array(mesas_values).reshape(len(mesas_values), -1)

    # Apply agglomerative clustering to group clients
    agg_cluster = AgglomerativeClustering(n_clusters=2)  # We assume 2 clusters (benign and malicious)
    agg_cluster.fit(mesas_values)

    # Identify the dominant cluster (the benign models)
    dominant_label = max(set(agg_cluster.labels_), key=list(agg_cluster.labels_).count)

    # Return indices of clients in the dominant (benign) cluster
    return np.where(agg_cluster.labels_ == dominant_label)[0]


def aggregate_updates(selected_updates, global_model):
    """
    Perform Federated Averaging (FedAvg) on the selected updates.

    Args:
    selected_updates: List of selected local updates for aggregation.
    global_model: The current global model.

    Returns:
    global_model: Updated global model after aggregation.
    """
    w_avg = copy.deepcopy(global_model)

    # Initialize the sum of updates
    for k in w_avg.keys():
        w_avg[k] = torch.zeros_like(global_model[k])

    # Sum the selected updates
    num_selected = len(selected_updates)
    for update in selected_updates:
        for k in update.keys():
            w_avg[k] += update[k]

    # Compute the average update
    for k in w_avg.keys():
        w_avg[k] /= num_selected

    return w_avg