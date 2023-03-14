
import numpy as np
import torch
import networkx as nx
import argparse
import random


import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch.autograd import Variable

from attention import MultiHeadSelfAttention

class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=False,
                 mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)

        nn.init.xavier_uniform_(
            self.W.weight,
            gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj):
        num_node_per_graph = adj.size(1)
        if self.use_bn and not hasattr(self, 'bn'):
            self.bn = nn.BatchNorm1d(num_node_per_graph).to(adj.device)

        if self.add_self:
            adj = adj + torch.eye(num_node_per_graph).to(adj.device)

        if self.mean:
            adj = adj / adj.sum(-1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k1 = self.W(h_k_N)
        h_k2 = F.normalize(h_k1, dim=1, p=2)
        h_k = F.relu(h_k2)

        return h_k

    def __repr__(self):
        if self.use_bn:
            return 'BN' + super(BatchedGraphSAGE, self).__repr__()
        else:
            return super(BatchedGraphSAGE, self).__repr__()



class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid):
        super(BatchedDiffPool, self).__init__()

        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)

        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        x_originalnumber = torch.matmul(s_l, xnext)

        return xnext, anext, x_originalnumber, s_l





class Attentional_Hierarchical_Graph(nn.Module):

    def __init__(self, hidden_dim, assign_dim):
        super(Attentional_Hierarchical_Graph, self).__init__()

        self.gc_before_pool = nn.ModuleList()
        self.diffpool_layers = nn.ModuleList()
        self.gc_after_pool = nn.ModuleList()
        self.assign_dim = assign_dim
        self.hidden_dim = hidden_dim

        self.gc_before_pool.append(
            BatchedGraphSAGE(self.hidden_dim, self.hidden_dim)
        )
        self.diffpool_layers.append(
            BatchedDiffPool(self.hidden_dim, self.assign_dim, self.hidden_dim),
        )
        self.gc_after_pool.append(
            BatchedGraphSAGE(self.hidden_dim, self.hidden_dim)
        )

        self.attention = MultiHeadSelfAttention(dim_in = self.hidden_dim, dim_k = self.hidden_dim, dim_v = self.hidden_dim, num_heads=1)

       # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data,
                                                     gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, adj, x):

        # out_all: store the results of each layer
        out_all = []

        # The first layer
        x = self.gc_before_pool[0](x, adj)
        out_all.append(x)

        # The second layer
        x_0, adj_0, x0_0, s1_0 = self.diffpool_layers[0](x, adj)
        out_all.append(x0_0)

        # The third layer
        x_1= self.gc_after_pool[0](x_0, adj_0)
        out_all.append(torch.matmul(s1_0, x_1))

        # Hierarchical attention fusion
        readout = torch.cat([out_all[0].unsqueeze(1),out_all[1].unsqueeze(1),out_all[2].unsqueeze(1)], dim=1)
        final_readout = self.attention(readout)
        final_readout = torch.sum(final_readout, 1)

        return final_readout