#!/usr/bin/env python
# encoding: utf-8
# File Name: gat.py
# Author: Jiezhong Qiu
# Create Time: 2017/12/18 21:40


# from __future__ import absolute_import
# from __future__ import unicode_literals
# from __future__ import division
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from public_gat_layers import BatchMultiHeadGraphAttention


class BatchGAT(nn.Module):
    def __init__(self,  # pretrained_emb,
                 n_units, n_heads,
                 dropout=0.1, attn_dropout=0.0, fine_tune=False,
                 instance_normalization=True):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units) - 1
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
        for i in range(self.n_layer):
            # consider multi head from last layer
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(n_heads[i], f_in=f_in,
                                             f_out=n_units[i + 1], attn_dropout=attn_dropout)
            )

    def forward(self, data, normalized_embedding=None):
        adj, x,  = data
        emb = normalized_embedding.float()
        x = torch.cat((x, emb), dim=2)
        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj)  # bs x n_head x n x f_out
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1)[:, -1, :]
