from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import copy
import torch
import torch.nn.functional as F
import numpy as np

def FLShield(w_list, global_model, dataset_test, args, w_updates, per_run, first_call, w_length, debug=True):
    """
    FLShield defense mechanism: Generates representative models, validates them, filters malicious models,
    clips, and performs aggregation to obtain the final global model update.
    """
    # Step 1: Cluster and generate representative models
    representative_models, client_cluster_map = cluster_and_average_representative_models(w_updates, args)

    # Step 2: Validate representative models using LIPC, passing the global_model and dataset_test
    validation_reports = validate_representative_models(representative_models, w_list, global_model, dataset_test, args)

    # Step 3: Filter representative models based on validation reports
    selected_models = filter_representative_models(validation_reports, args)
    ##
    #print('selected_models: ', selected_models)
    # Step 4: Perform clipping on selected local updates
    #selected_indices = [i for i in range(len(representative_models)) if representative_models[i] in selected_models]
    selected_indices = [
        client_idx
        for cluster_id in selected_models
        for client_idx in client_cluster_map[cluster_id]
    ]
    ##
    #print('selected_indices: ', selected_indices)
    selected_length = [w_length[i] for i in selected_indices]
    selected_length = [i / sum(selected_length) for i in selected_length]
    total_clients_this_round = len(w_list)
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
    mode = "w" if first_call else "a"
    if debug:
        filename = './' + args.save + '/' + args.dataset + '/' + args.iid + '/Flshield_analysis_{}_frac={}_nattacker={}_epsilon_{}_clip_{}_lr_{}_round_{}.txt'.format(
            args.attack_type, args.frac, args.num_attacker, str(args.dp_epsilon), str(args.dp_clip), str(args.lr),
            per_run)
        with open(filename, mode) as f:
            f.write(f"malicious_rate: {malicious_rate:.4f}\n")
            f.write(f"benign_rate:    {benign_rate:.4f}\n")
            f.write("--------Round--------\n")

    clipped_updates = clipping_local_updates_med(w_updates, selected_indices, args)

    # Step 5: Aggregate the clipped updates to obtain the final global model
    aggregated_updates = aggregate_updates(global_model.state_dict(), clipped_updates, selected_length)

    central_param = global_model.state_dict()
    w_avg = {}
    for key in central_param.keys():
        w_avg[key] = central_param[key] + aggregated_updates[key]

    return w_avg


def cluster_and_average_representative_models(w_updates, args):
    """
    Cluster local updates into groups and average the updates in each cluster to form representative models.
    Returns the representative models and client-cluster mapping.
    """
    updates_vector = [parameters_dict_to_vector_flt(update).cpu().numpy() for update in w_updates]

    # Define variables to track the best clustering
    optimal_clusters = 2  # Start with the minimum number of clusters
    best_score = -1
    best_kmeans = None

    # Dynamically try different numbers of clusters based on silhouette score
    for num_clusters in range(2, len(w_updates)):  # Iterate over possible number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(updates_vector)
        score = silhouette_score(updates_vector, kmeans.labels_)

        # If we find a better silhouette score, update the optimal clusters and best score
        if score > best_score and score > args.silhouette_threshold:
            best_score = score
            optimal_clusters = num_clusters
            best_kmeans = kmeans

    # If no optimal clustering found, fallback to the predefined number of clusters
    if best_kmeans is None:
        ##
        #print("Fallback: Using predefined number of clusters")
        num_clusters = args.num_clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(updates_vector)
    else:
        ##
        #print(f"Optimal number of clusters found: {optimal_clusters} with Silhouette Score: {best_score}")
        kmeans = best_kmeans  # Use the best clustering result

    # Generate representative models by averaging updates in each cluster
    representative_models = []
    client_cluster_map = {}

    for i, label in enumerate(kmeans.labels_):
        if label not in client_cluster_map:
            client_cluster_map[label] = []
        client_cluster_map[label].append(i)

    #for cluster_idx, updates_in_cluster in client_cluster_map.items():
    for cluster_id, client_idx_list in client_cluster_map.items():
        updates_in_cluster = [w_updates[j] for j in client_idx_list]
        # Average the updates within the cluster to form the representative model
        avg_weights = average_model_updates(updates_in_cluster)
        stripped = {}
        for k, v in avg_weights.items():

            if k.startswith("_module."):
                new_key = k.replace("_module.", "")
            else:
                new_key = k
            stripped[new_key] = v
        # Apply the averaged weights to a new instance of the model
        rep_model = create_new_model_instance(args)  # Create a new model instance
        rep_model.load_state_dict(stripped)  # Load the averaged weights into the model

        representative_models.append(rep_model)

    return representative_models, client_cluster_map


def validate_representative_models(representative_models, w_updates, global_model, dataset_test, args):
    """
    Validators evaluate the representative models and return validation reports.
    """
    validation_reports = []
    validators = random.sample(range(len(w_updates)), args.num_validators)

    for model in representative_models:
        report = []
        for validator in validators:
            local_model = w_updates[validator]
            validation_loss = compute_validation_loss(local_model, model, global_model, dataset_test, args)
            report.append(validation_loss)

        validation_reports.append(report)

    return validation_reports


def parameters_dict_to_vector_flt(net_dict) -> torch.Tensor:
    vec = []
    for key, param in net_dict.items():
        # print(key, torch.max(param))
        if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def average_model_updates(model_updates):
    # Make a deep copy of the first model as the base for averaging
    avg_model = copy.deepcopy(model_updates[0])

    # Initialize the avg_model to zero
    for k in avg_model.keys():
        avg_model[k] = torch.zeros_like(model_updates[0][k])

    # Add up all the model updates in the cluster
    num_models = len(model_updates)
    for update in model_updates:
        for k in avg_model.keys():
            avg_model[k] += update[k]

    # Divide by the number of models to get the average
    for k in avg_model.keys():
        avg_model[k] = avg_model[k] / num_models

    return avg_model

from models.Nets import CNNMnist, CNNCifar_ResNet18, FastTextBinary
def create_new_model_instance(args):
    """
    This function returns a new instance of the model architecture.
    Replace `Net` with your actual model class.
    """
    if args.dataset in ('mnist', 'fashion-mnist'):
        model = CNNMnist(args)
    elif args.dataset == 'cifar':
        model = CNNCifar_ResNet18(args)
    elif args.dataset == 'sent140':
        model = FastTextBinary(args.vocabSize)
    else:
        print('No vaild model!!======')
        exit(0)
    model = model.to(args.device)  # Move the model to the correct device (GPU or CPU)
    return model

def compute_validation_loss(local_model, representative_model, global_model, dataset_test, args, num_samples=None):
    """
    Compute the validation loss between a local model and a representative model using LIPC (loss impact per class).
    """
    # Load validation data
    validation_data = get_validation_data(dataset_test, num_samples)

    # Initialize loss vectors for L(E,v) and move it to the correct device
    classwise_loss_vector = torch.zeros(args.num_classes).to(args.device)
    with torch.no_grad():
        for x, y in validation_data:
            x = x.to(args.device)  # Move input to the correct device
            y = y.to(args.device)  # Move labels to the correct device

            logits_global = global_model(x)
            logits_representative = representative_model(x)

            # Loop over each sample in the batch
            for i in range(len(y)):
                current_class = y[i].item()  # Get the class label for the current sample
                for class_idx in range(args.num_classes):
                    if current_class == class_idx:
                        # Loss for global model and representative model for the current sample
                        loss_global = F.cross_entropy(logits_global[i:i+1], y[i:i+1])
                        loss_rep = F.cross_entropy(logits_representative[i:i+1], y[i:i+1])

                        # Compute loss impact (difference between losses)
                        loss_impact = (loss_global - loss_rep).to(args.device)  # Ensure loss_impact is on the correct device

                        # Accumulate the loss impact for the current class
                        classwise_loss_vector[class_idx] += loss_impact

    # Average the loss across all samples for each class
    classwise_loss_vector /= len(validation_data)

    return classwise_loss_vector


def get_validation_data(dataset_test, num_samples=None):
    """
    Fetch the validation data for the validator client from the dataset_test.
    If num_samples is specified, only return a subset of the validation data.
    """
    if num_samples is not None:
        # Sample a subset of the test dataset
        indices = torch.randperm(len(dataset_test))[:num_samples]
        validation_subset = torch.utils.data.Subset(dataset_test, indices)
        validation_loader = torch.utils.data.DataLoader(
            validation_subset, batch_size=64, shuffle=False)  # Batch size can be set here or passed via args
    else:
        # Use the full test dataset
        validation_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=64, shuffle=False)

    return validation_loader

def filter_representative_models(validation_reports, args):
    """
    Filter representative models based on the validation reports.
    Select the top 50% models based on LIPC scores across all classes.
    """

    loss_list = []
    # Compute average validation loss for each representative model
    for idx, report in enumerate(validation_reports):
        # Flatten the class-wise loss vectors and find the minimum value
        avg_loss_vector = torch.mean(torch.stack(report), dim=0)
        min_loss_value = torch.min(avg_loss_vector)
        loss_list.append((idx, min_loss_value.item()))

    # Select the top 50% of the representative models
    loss_list.sort(key=lambda x: x[1])
    num_models = len(loss_list)
    topk = num_models // 2
    top50_idx = [loss_list[i][0] for i in range(topk)]

    return top50_idx

def clipping_local_updates(w_updates, selected_indices, args):
    """
    Perform clipping on selected local updates to ensure no update disproportionately affects the global model.
    """
    clipped_updates = []

    for idx in selected_indices:
        update = w_updates[idx]
        clipped_update = {key: torch.clamp(val, -args.clip_value, args.clip_value) for key, val in update.items()}
        clipped_updates.append(clipped_update)

    return clipped_updates

def clipping_local_updates_med(w_updates, selected_indices, args):
    """
    Perform clipping on selected local updates based on the median to ensure no update disproportionately affects the global model.
    """

    norms = []
    for idx in selected_indices:
        upd_dict = w_updates[idx]
        vec = parameters_dict_to_vector_flt(upd_dict).to(args.device if hasattr(args, 'device') else vec.device)
        norm_val = torch.norm(vec, p=2).item()
        norms.append(norm_val)

    clip_med = float(np.median(norms))

    clipped_updates = []
    for idx, norm_val in zip(selected_indices, norms):
        update = w_updates[idx]

        if norm_val <= clip_med:
            clipped_updates.append(copy.deepcopy(update))
        else:

            gamma = clip_med / norm_val  # gamma < 1

            scaled_update = {}
            for key, tensor in update.items():

                if key.split('.')[-1] in ('num_batches_tracked', 'running_mean', 'running_var'):
                    scaled_update[key] = tensor.clone()
                else:
                    scaled_update[key] = tensor.clone() * gamma
            clipped_updates.append(scaled_update)

    return clipped_updates


def aggregate_updates(global_model, selected_updates, weight):
    """
    Perform Federated Averaging (FedAvg) on the selected updates.

    Args:
    selected_updates: List of selected local updates for aggregation.
    global_model: The current global model.

    Returns:
    global_model: Updated global model after aggregation.
    """
    aggregated_update = {k: torch.zeros_like(global_model[k]) for k in global_model.keys()}

    # Sum the selected updates
    num_selected = len(selected_updates)
    for update, weight in zip(selected_updates, weight):
        for k in update.keys():
            aggregated_update[k] += update[k] * weight

    return aggregated_update