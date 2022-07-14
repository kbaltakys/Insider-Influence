import os
from collections import Counter
import re
from typing import Iterable

import networkx as nx
import torch
from torch.utils.data import DataLoader, Subset
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm

from analysistools.parse_event_hparams import read_hyper_params
from datatools.data_loader import return_train_valid_test_sets
from interface import Parameters
import numpy as np
import json

from models.gat import BatchGAT
from models.gcn import BatchGCN
from train import evaluate
from datatools.parameters import SECURITY_LIST

"""
1. Load the appropriate data set
2. Select the test data set portion

"""


def load_full_network(adjlist_file: str):
    g = nx.read_adjlist(adjlist_file, nodetype=int)
    degrees = dict(g.degree())
    return degrees


def bin_loader(measure: Iterable, n_bins: int, loader: DataLoader):
    test_sample_indexes = loader.sampler.index
    sample_egos = list()
    for i_batch, (data, target) in enumerate(loader):
        graphs, influence_features, vertices = data
        sample_egos.append(vertices[:, -1].numpy())
    sample_egos = np.hstack(sample_egos)
    sample_ego_measures = np.array([measure[ego] for ego in sample_egos])

    quantiles = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(sample_ego_measures, q=quantiles)
    if len(np.unique(bins)) <= n_bins:
        raise Exception(f'Cannot divide data into {n_bins} equal parts.')
    bins[0] = sample_ego_measures.min() - 1
    bin_assignment = np.digitize(sample_ego_measures, bins, right=True)
    bin_intervals = zip(bins[:-1], bins[1:])
    measure_bin_loaders = {}
    for bin_id, interval in zip(range(1, n_bins + 1), bin_intervals):
        bin_sample_ids = test_sample_indexes[bin_assignment == bin_id]
        bin_subset = Subset(loader.dataset, indices=bin_sample_ids)
        bin_loader = DataLoader(bin_subset, batch_size=64)
        measure_bin_loaders[interval] = bin_loader

    return measure_bin_loaders


def get_model_predictions(
    #   path_data_root: str,
    path_model: str,
    #   path_result: str,
    #   dataset: str,
    #   DEVICE
    device: str = 'gpu'
):

    if 'real_0' in path_model:
        # print('Dealing with real network data')
        network_type = 'real_0'
        path_data_root = f'./data/real_data/{network_type}/'
    elif 'random_' in path_model:
        # print('Dealing with random network data')
        network_type = re.findall('(?<=/)[a-z]+_[0-9]+', path_model)[0]
        path_data_root = f'./data/random/{network_type}/'
    else:
        raise Exception('Unkown input data used')

    dataset = re.findall('(?<=/)[a-zA-Z\-]+_[DW]_(?:Sell|Buy)', path_model)
    if dataset:
        dataset = dataset[0]
    else:
        raise Exception('Unknown dataset')

    # Setup is here
    file_dir = os.path.join(path_data_root, 'derivatives')
    data_path = os.path.join(file_dir, dataset)
    path_model_checkpoint = os.path.join(path_model, 'checkpoint.pt')

    path_model_parameter_summary = [
        os.path.join(path_model, possible_folder)
        for possible_folder in os.listdir(path_model)
        if os.path.isdir(os.path.join(path_model, possible_folder))][0]

    path_model_parameter_summary = os.path.join(
        path_model_parameter_summary,
        os.listdir(path_model_parameter_summary)[0])

    hparams = read_hyper_params(path_model_parameter_summary)

    arguments = [
        '--result-dir', './results/model_reevaluation_trash',
        '--file-dir', data_path,
        '--model', hparams['model'],
        '--dropout', str(hparams['drop-out']),
        '--data-split-seed', str(int(hparams['data-split-seed'])),
        '--seed', str(int(hparams['seed']))]

    if device == 'cpu':
        arguments.extend(['--no-cuda'])
        device = torch.device("cpu")
    elif device == 'gpu':
        device = torch.device("cuda:1")

    if hparams['model'] == 'gat':
        hidden_units, heads = hparams['hidden-units'].split(' heads-')
        arguments.extend(['--hidden-units', hidden_units,
                          '--heads', heads])
    elif hparams['model'] == 'gcn':
        hidden_units = hparams['hidden-units']
        arguments.extend(['--hidden-units', hidden_units])
    else:
        raise NotImplementedError('Unknown model')

    args = Parameters(arguments).args

    data_loader = return_train_valid_test_sets(args)

    test_loader = data_loader['test']
    test_loader.sampler.shuffle = False
    data_loader['valid'].sampler.shuffle = False

    path_node_mapping = os.path.join(path_data_root, 'node_labels.json')
    with open(path_node_mapping, 'r') as node_labels:
        owner_map = json.load(node_labels)
    owner_map = {int(owner): owner_map[owner] for owner in owner_map}

    security_ids = np.load(os.path.join(data_path, "security_ids.npy"))
    security_map = dict(enumerate(SECURITY_LIST))
    test_index = test_loader.dataset.test_index
    sample_securities = [security_map[security_id] for security_id in
                         security_ids[test_index]]

    sample_egos = list()
    for data, _ in test_loader:
        _, _, vertices = data
        sample_egos.append(vertices[:, -1].numpy())
    sample_egos = np.hstack(sample_egos)
    sample_owners = [owner_map[ego] for ego in sample_egos]

    embedding = data_loader['test'].dataset.get_embedding()
    vertex_features = data_loader['test'].dataset.get_vertex_features()
    n_neighbors = data_loader['test'].dataset.n_neighbors

    feature_dim = data_loader['test'].dataset.get_feature_dimension()
    n_units = [feature_dim] + \
              [int(x) for x in args.hidden_units.strip().split(",")] + [
                  data_loader['test'].dataset.n_classes]

    if args.class_weight_balanced:
        class_weight = test_loader.dataset.get_class_weight()
    else:
        class_weight = torch.ones(test_loader.dataset.n_classes)

    if args.model == "gcn":
        model = BatchGCN(pretrained_emb=embedding,
                         vertex_feature=vertex_features,
                         use_vertex_feature=args.use_vertex_feature,
                         n_neighbors=n_neighbors,
                         n_units=n_units,
                         dropout=args.dropout,
                         instance_normalization=args.instance_normalization,
                         last_conversion=args.last_conversion)
    elif args.model == "gat":
        n_heads = [int(x) for x in args.heads.strip().split(",")]
        model = BatchGAT(pretrained_emb=embedding,
                         vertex_feature=vertex_features,
                         use_vertex_feature=args.use_vertex_feature,
                         n_units=n_units, n_heads=n_heads,
                         dropout=args.dropout,
                         instance_normalization=args.instance_normalization)

    model.load_state_dict(
        torch.load(path_model_checkpoint))  # , map_location=torch.device('cuda:1')
    model.to(device)
    model.eval()
    valid_loss, best_thr, valid_stats = evaluate(model, args, class_weight,
                                                 data_loader['valid'])
    assert hparams['best_threshold'] == best_thr, \
        print(
            f'Best thresholds do not match {hparams["best_threshold"]} vs {best_thr}')

    _, _, stats = evaluate(model, args, class_weight,
                           test_loader, best_thr=best_thr)

    df = pd.DataFrame(
        [sample_egos, sample_owners, sample_securities,
         stats['predicted_labels'], stats['labels'].numpy(),
         stats['predictions'].numpy()],
        index=['ego_id', 'owner_id', 'security', 'prediction', 'label',
               'score']).T
    df['best_thr'] = np.exp(best_thr)
    df['dataset'] = dataset
    df['dataset_seed'] = args.data_split_seed
    df['seed'] = args.seed
    df['network_type'] = network_type
    return df


def retrieve_model_predictions(dirpaths):
    model_predictions = []
    progress_bar = tqdm(dirpaths, desc='Retrieving model predictions')
    for model_path in progress_bar:
        df = get_model_predictions(model_path)
        model_predictions.append(df)
    mp = pd.concat(model_predictions, ignore_index=True)
    mp = mp.astype(dtype={'ego_id': int, 'owner_id': int, 'security': str,
                          'prediction': int, 'label': int, 'score': float,
                          'best_thr': float, 'dataset': str, 'dataset_seed': int,
                          'seed': int, 'network_type': str})
    return mp


def save_model_predictions(path_to_models: str):
    model_dirpaths = []
    for dirpath, dirnames, filenames in os.walk(path_to_models):
        if 'checkpoint.pt' in filenames:
            model_dirpaths.append(dirpath)

    M = retrieve_model_predictions(model_dirpaths)
    M.to_hdf(os.path.join(path_to_models, 'predictions.hdf'), 'table')



if __name__ == '__main__':

    paths_to_models = [
        './results/random_data_splits_for_best_dataset_models_fixed_threshold_fixed'
        ]

    for path_to_models in paths_to_models:
       save_model_predictions(path_to_models)