import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GTN.layers import GraphConvolution
import scipy.sparse as sp
import main_gcn_trans

class GTN(nn.Module):
    """
    Using Transformer on Graph Convolutional Networks for Node Embedding
    """

    def __init__(self, in_dim=1, hidden_dim=16, out_dim=1, n_head=1, dropout=0.1, num_GC_layers=1):
        super(GTN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.num_GC_layers = num_GC_layers
        if self.num_GC_layers == 1:
            self.gc1 = GraphConvolution(in_dim, out_dim)
        if self.num_GC_layers == 2:
            self.gc1 = GraphConvolution(in_dim, hidden_dim)
            self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.fc = nn.Linear(out_dim * 2, out_dim)
        encoder_transformer_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=n_head, dim_feedforward=hidden_dim,
                                                               dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_transformer_layer, num_layers=1)
    def normalize(A):
        rowsum=np.array(A.sum(1))
        r_inv=np.power(rowsum,-1).flatten()
        r_inv[np.isinf(r_inv)]=0
        r_mat_inv=sp.diags(r_inv)
        x=r_mat_inv.dot(A)
        return x
    def forward(self, x, A, nodes_u, nodes_v):
        x=F.normalize(A)
        h = F.relu(self.gc1(x, A))
        h = F.dropout(h, self.dropout, training=self.training)
        if self.num_GC_layers == 2:
            h = F.relu(self.gc2(h, A))
            h = F.dropout(h, self.dropout, training=self.training)
        h = h.view(-1, 1, self.out_dim)
        h = self.transformer(h)
        h = h.view(-1, self.out_dim)
        h_uv = torch.cat((h[nodes_u], h[nodes_v]), 1)
        scores = self.fc(h_uv)
        return scores
