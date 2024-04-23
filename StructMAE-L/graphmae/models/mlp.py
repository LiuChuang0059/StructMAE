import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from graphmae.utils import create_activation, NormLayer, create_norm

class MLP(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout, activation, norm, encoding = False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.head = nn.Identity()
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear_or_not = True
            self.linear = nn.Linear(in_dim, out_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(in_dim, num_hidden))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(num_hidden, num_hidden))
            self.linears.append(nn.Linear(num_hidden, out_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(num_hidden))
                self.activations.append(create_activation(activation))

    def forward(self, x, edge_index, return_hidden=False):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)
