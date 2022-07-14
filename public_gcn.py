#!/usr/bin/env python
# encoding: utf-8
# File Name: gcn.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/17 14:11


# from __future__ import absolute_import
# from __future__ import unicode_literals
# from __future__ import division
# from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from public_gcn_layers import BatchGraphConvolution
from torch.nn.parameter import Parameter
import torch.nn.init as init


class BatchGCN(nn.Module):
    def __init__(self, n_units, dropout,  # pretrained_emb,
                 n_neighbors, fine_tune=False,
                 instance_normalization=True, last_conversion=False):
        super(BatchGCN, self).__init__()
        self.num_layer = len(n_units) - 1
        self.last_conversion = last_conversion
        self.n_labels = n_units[0] - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(64,  # pretrained_emb.size(1),
                                          momentum=0.0, affine=True)

        # https://discuss.pytorch.org/t/can-we-use-pre-trained-word-embeddings-for-weight-initialization-in-nn-embedding/1222/2
        # For the public data this is not necessary to train, but since our
        # models were trained with non-public data this is left here.
        # 1790 is the total number of nodes in the network.
        self.embedding = nn.Embedding(1790, 64)
        self.embedding.weight.requires_grad = fine_tune
        n_units[0] += 64

        self.layer_stack = nn.ModuleList()

        for i in range(self.num_layer):
            self.layer_stack.append(
                BatchGraphConvolution(n_units[i], n_units[i + 1])
            )
        if self.last_conversion:
            self.last_weights = Parameter(torch.Tensor(
                self.n_labels, n_neighbors))
            init.xavier_uniform_(self.last_weights)

    def forward(self, data, normalized_embedding=None):
        lap, x = data
        lap = lap.float()
        emb = normalized_embedding.float()
        x = torch.cat((x, emb), dim=2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, lap)
            if i + 1 < self.num_layer + self.last_conversion:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        if self.last_conversion:
            expand_weight3 = self.last_weights.expand(
                x.shape[0], -1, -1)
            x = torch.bmm(expand_weight3, x)
        return F.log_softmax(x, dim=-1)[:, -1, :]
