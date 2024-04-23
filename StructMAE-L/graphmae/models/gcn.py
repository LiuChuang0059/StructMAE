import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from graphmae.utils import create_activation


class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        self.norms = None
        self.head = nn.Identity()

    def forward(self, inputs, edge_index, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](h, edge_index)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GraphConv(MessagePassing):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True):
        # super(GraphConv, self).__init__(aggr='add')
        super().__init__(aggr='add') # 聚合方式：求和
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self.in_dim != self.out_dim:
                self.res_fc = nn.Linear(self.in_dim, self.out_dim, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        # Add self-loops and calculate degree for normalization
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg = degree(edge_index[0], dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        # Linear transformation
        x = self.fc(x)

        # Message passing
        x = self.propagate(edge_index, x=x, norm=norm)

        # Residual connection
        if self.res_fc is not None:
            x = x + self.res_fc(x)

        # Normalization
        if self.norm is not None:
            x = self.norm(x)

        # Activation function
        if self.activation is not None:
            x = self.activation(x)

        return x

    def message(self, x_j, norm):
        # Message function for aggregation
        return norm.view(-1, 1) * x_j