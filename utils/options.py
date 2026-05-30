#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=120, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--alr', type=float, default=0.1, help="attacker learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', type=str, help='iid or qty or dir...')
    parser.add_argument('--alpha', type=float, default=0.5, help="level of non-iid in dirichlet_distribution_label_imbalance,used in Moon,etc.")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--attack', action='store_true', help='backdoor attack')
    parser.add_argument('--backdoor_baseline', type=str, default='standard',
                        choices=['standard', 'DBA', 'Neurotoxin'],
                        help='backdoor baseline: standard, DBA, or Neurotoxin')
    parser.add_argument('--num_attacker', type=int, default=1, help='number of attacker (default: 1)')
    parser.add_argument('--dp_mechanism', type=str, default='Gaussian',
                        help='differential privacy mechanism')
    parser.add_argument('--dp_epsilon', type=float, default=20,
                        help='differential privacy epsilon')
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help='differential privacy delta')
    parser.add_argument('--dp_clip', type=float, default=10,
                        help='differential privacy clip')
    parser.add_argument('--dp_sample', type=float, default=1, help='sample rate for moment account')

    parser.add_argument('--serial', action='store_true', help='partial serial running to save the gpu memory')
    parser.add_argument('--serial_bs', type=int, default=128, help='partial serial running batch size')
    parser.add_argument('--dim', type=int, default=28, help='noise dataset image size (same as training image)')
    parser.add_argument('--clipping_thresholds', type=float, default=1 / 3, help="clipping thresholds of update norm")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_samples', type=int, default=20000, help="number of samples of noise dataset")
    parser.add_argument('--attack_type', type=str, default='Collusion', help="Specific the attackers' behavior (Collusion, Input, Output)")
    parser.add_argument('--defense', type=str, help="Defense method")
    parser.add_argument('--log_distance', type=bool, default=False, help="output krum distance")
    parser.add_argument('--PDR', type=float, default=1, help="poison data rate")
    parser.add_argument('--neuro_mask_ratio', type=float, default=0.95,
                        help='fraction of small-gradient parameters kept by Neurotoxin')
    parser.add_argument('--neuro_aggregate_all_layer', type=int, default=0,
                        help='use one global Neurotoxin mask threshold when set to 1')
    parser.add_argument('--flame_noise', type=float, default=0.001, help='Flame noise')
    parser.add_argument('--save', type=str, default='save', help="dic to save results (ending without /)")
    parser.add_argument('--silhouette_threshold', type=float, default=0.5, help="Minimum silhouette score for clustering.")
    parser.add_argument('--num_clusters', type=int, default=5, help="Number of clusters for K-means in FLShield.")
    parser.add_argument('--num_validators', type=int, default=5, help="Number of validators in FLShield.")
    parser.add_argument('--loss_threshold', type=float, default=0.1, help="Loss threshold for selecting representative models.")
    parser.add_argument('--clip_value', type=float, default=1.0, help="Clipping value for bounding model updates.")
    parser.add_argument('--server_dataset', type=int, default=100, help="number of dataset in server")
    parser.add_argument('--plr_class', type=int, default=6, help="get PLRs of a specific class")
    parser.add_argument('--tau', type=float, default=0.8, help="threshold of LPA_ER")
    parser.add_argument('--thre_labels', type=float, default=2, help="threhold of labels per client in  quantity_based_non-iid_label_imbalance,used in FedAvg,etc.")
    parser.add_argument('--total_run', type=int, default=5, help="how many rounds totally run")
    parser.add_argument('--round_indicator', type=int, default=0, help="round number to start")
    parser.add_argument('--re_weight', action='store_true', help="use the dataset size to reweight the update")
    parser.add_argument('--vocabSize', type=int, default=0, help="vocabsize for sent140")
    parser.add_argument('--central_noise', type=float, default=0, help="std of the central noise")
    parser.add_argument('--random_drop', type=float, default=0, help="std of the central noise")
    args = parser.parse_args()
    return args
