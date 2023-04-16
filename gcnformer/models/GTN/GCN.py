import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GTN.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_GC_layers=2):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_GC_layers = num_GC_layers
        self.dropout = dropout
        if self.num_GC_layers == 1:
            self.gc1 = GraphConvolution(in_dim, out_dim)
        if self.num_GC_layers == 2:
            self.gc1 = GraphConvolution(in_dim, hidden_dim)
            self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.fc = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x, adj, nodes_u, nodes_v):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        if self.num_GC_layers == 2:
            h = F.relu(self.gc2(h, adj))
            h = F.dropout(h, self.dropout, training=self.training)
        h_uv = torch.cat((h[nodes_u], h[nodes_v]), 1)
        scores = self.fc(h_uv)
        return scores


