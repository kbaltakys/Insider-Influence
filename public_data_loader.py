#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 20:28:44 2022

@author: baltakys
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import logging

# from datatools.data_preprocessing import DataProcessor

logger = logging.getLogger(__name__)


class ChunkSampler(Sampler):

    def __init__(self, index, shuffle=False):
        """
            Parameters
            ----------
            index : np.array
                index of elements of the data set
            shuffle : bool
                if True the
        """
        self.shuffle = shuffle
        self.num_samples = len(index)
        self.index = index

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.index)
        return iter(self.index)

    def __len__(self):
        return self.num_samples


def stratified_split(labels: np.array, fr_train: float, fr_valid: float,
                     shuffle: bool = True):
    """
    Create data split into train, validation and test data sets so that the
    fraction of positive labels in each data set is the same.

    Parameters
    ----------
    labels : np.array
        data set labels.
    fr_train : float
        fraction of data set dedicated for training.
    fr_valid : float
        fraction of data set dedicated for validation.
    fr_test : float
        fraction of data set dedicated for testing.
    shuffle : bool, optional
        If the data set should be shuffled. If True, train/valid/test sets can
        have data from any part of the sample set.
        The default is True.

    Returns
    -------
    data_split : TYPE
        DESCRIPTION.

    """
    data_split = np.zeros_like(labels)
    classes = np.unique(labels)
    for class_ in classes:
        class_ids = np.argwhere(labels == class_).squeeze()
        if shuffle:
            np.random.shuffle(class_ids)
        class_size = len(class_ids)
        valid_idx = class_ids[int(class_size * fr_train):
                              int(class_size * (fr_valid + fr_train))]
        data_split[valid_idx] = 1
        test_idx = class_ids[int(class_size * (fr_valid + fr_train)):]
        data_split[test_idx] = 2
    return data_split


class AnonymizedInfluenceDataSet(Dataset):
    def __init__(self, args):
        self.args = args
        self.public_file_dir = args['public_file_dir']

        self.train_ratio = args['train_ratio']
        self.valid_ratio = args['valid_ratio']
        self.model = args['model']

        self.dataset_split = None
        self.train_index = None
        self.valid_index = None
        self.test_index = None

        graphs = pd.read_csv(self.public_file_dir + 'Public_Graphs.csv')
        graphs.set_index(['observation id', 'neighbor id'], inplace=True)
        self.graphs = np.array(
            [graph.values for _, graph in graphs.groupby(level=['observation id'])])
        self.graphs = self.prepare_graphs(graphs=self.graphs)
        print("graphs loaded and prepared! Shape: {}".format(self.graphs.shape))

        influence_features = pd.read_csv(
            self.public_file_dir + 'Public_Influence_Features.csv')
        influence_features.set_index(['observation id', 'neighbor id'], inplace=True)

        self.influence_features = np.array(
            [influence_features.values for _, influence_features in influence_features.groupby(level=['observation id'])])
        print("influence features loaded! Shape: {}".format(
            self.influence_features.shape))

        labels = pd.read_csv(self.public_file_dir + 'Public_Labels.csv', index_col=[0])
        self.labels = labels.label.values
        print("labels loaded! Shape: {}".format(self.labels.shape))

        distances = pd.read_csv(
            self.public_file_dir + 'Public_Distances.csv', index_col=[0])
        self.distances = distances.distance.values
        print("distances loaded! Shape: {}".format(self.distances.shape))

        family_flags = pd.read_csv(
            self.public_file_dir + 'Public_Family_flag.csv', index_col=[0])
        self.family_flags = family_flags.family_flag.values
        print("family_flags loaded! Shape: {}".format(self.family_flags.shape))

        own_company_flags = pd.read_csv(
            self.public_file_dir + 'Public_Own_Company_flag.csv', index_col=[0])
        self.own_company_flags = own_company_flags.own_company.values
        print("own_company_flags loaded! Shape: {}".format(
            self.own_company_flags.shape))

        normalized_embeddings = pd.read_csv(
            self.public_file_dir + 'Public_Normalized_Embedding.csv')

        normalized_embeddings.set_index(['observation id', 'neighbor id'], inplace=True)
        self.normalized_embeddings = np.array(
            [normalized_embedings.values for _, normalized_embedings in normalized_embeddings.groupby(level=['observation id'])])
        # self.normalized_embeddings = np.load(self.public_file_dir + 'Public_Normalized_Embedding.npy')
        # self.normalized_embeddings = torch.load(self.public_file_dir + 'Public_Normalized_Embedding_tensor.pt')
        print("normalized_embeddings loaded! Shape: {}".format(
            self.normalized_embeddings.shape))

        self.N = self.graphs.shape[0]
        print("{} ego networks loaded, each with size {}".format(
            self.N, self.graphs.shape[1]))

        self.n_classes = self.get_num_class()
        print('Number of classes: {}'.format(self.n_classes))

        class_weight = self.N / (self.n_classes * np.bincount(self.labels))
        self.class_weight = torch.FloatTensor(class_weight)

        dataset_split = pd.read_csv(
            self.public_file_dir + 'Public_Dataset_splits.csv', index_col=[0])
        self.dataset_split = dataset_split.data_split.values

        self.train_index = np.argwhere(self.dataset_split == 0).squeeze()
        self.valid_index = np.argwhere(self.dataset_split == 1).squeeze()
        self.test_index = np.argwhere(self.dataset_split == 2).squeeze()


    def prepare_graphs(self, graphs):
        # self-loop trick, the input graphs should have no self-loop
        self.n_neighbors = graphs.shape[1]
        identity = np.identity(self.n_neighbors, dtype=int)
        graphs += identity
        graphs[graphs != 0] = 1.0
        if self.model == "gat":
            graphs = graphs.astype(np.dtype('B'))
        elif self.model == "gcn":
            # normalized graph laplacian for GCN: D^{-1/2}AD^{-1/2}
            graphs = graphs.astype(float)
            for i in range(len(graphs)):
                graph = graphs[i]
                d_root_inv = 1. / np.sqrt(np.sum(graph, axis=1))
                graph = (graph.T * d_root_inv).T * d_root_inv
                graphs[i] = graph
        else:
            raise NotImplementedError
        return graphs

    def get_feature_dimension(self):
        return self.influence_features.shape[-1]

    def get_num_class(self):
        return np.unique(self.labels).shape[0]

    def get_class_weight(self):
        return self.class_weight

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (self.graphs[idx], self.influence_features[idx],
                self.distances[idx], self.family_flags[idx],
                self.own_company_flags[idx],
                self.normalized_embeddings[idx]), self.labels[idx]



def return_train_valid_test_sets(args: dict, batch_size: int, shuffle: bool = True):
    influence_dataset = AnonymizedInfluenceDataSet(args)

    train_loader = DataLoader(
        influence_dataset, batch_size=batch_size,
        sampler=ChunkSampler(influence_dataset.train_index, shuffle=shuffle)
    )
    valid_loader = DataLoader(
        influence_dataset, batch_size=batch_size,
        sampler=ChunkSampler(influence_dataset.valid_index, shuffle=shuffle)
    )
    test_loader = DataLoader(
        influence_dataset, batch_size=batch_size,
        sampler=ChunkSampler(influence_dataset.test_index, shuffle=shuffle)
    )
    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader}
