import torch
import torch.nn as nn
import torch.nn.functional as F
from model.mmpgnet.utils import get_adj_tensor, get_normalize_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge, drop_feature, kaiming_uniform
from torch_geometric.utils import degree
from utils import set_seed
set_seed()

EDGEGCN_DROPOUT_P = 0.2
print("EdgeGCN default dropout_p={}".format(EDGEGCN_DROPOUT_P))


class EdgeGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(EdgeGCNLayer, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_transform = nn.Linear(edge_dim, out_channels)

    def forward(self, x, edge_index, edge_features):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_features: [num_edges, edge_dim]
        """
        # 转换节点特征
        x = self.lin(x)  # [num_nodes, out_channels]

        # 转换edge feature
        edge_features = self.edge_transform(edge_features)  # [num_edges, out_channels]

        row, col = edge_index

        # 1. Compute the degree of each node
        #    Note: col represents the target node. For undirected graphs, in-degree equals out-degree
        deg = degree(col, x.size(0), dtype=x.dtype)

        # 2. Add a small epsilon to each degree to prevent division by zero
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 3. Compute normalized weights for each edge: norm[e] = 1 / sqrt(deg[i] * deg[j])
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # ------------------------------------

        # Multiply source node features with corresponding edge features
        messages = x[row] * edge_features  # [num_edges, out_channels]

        # Scale messages by their normalized edge weights
        messages = norm.view(-1, 1) * messages

        # Use scatter_add to efficiently aggregate messages to target nodes
        out = torch.zeros_like(x)
        out.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(-1)), messages)

        return out


class EdgeGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, dropout=EDGEGCN_DROPOUT_P):
        super(EdgeGCN, self).__init__()

        self.num_layers = num_layers

        self.edge_weight = nn.Parameter(torch.Tensor(edge_dim, edge_dim))
        self.edge_bias = nn.Parameter(torch.Tensor(edge_dim))
        self.layers = nn.ModuleList()
        self.layers.append(EdgeGCNLayer(in_channels, hidden_channels, edge_dim))
        for _ in range(num_layers - 2):
            self.layers.append(EdgeGCNLayer(hidden_channels, hidden_channels, edge_dim))

        self.layers.append(EdgeGCNLayer(hidden_channels, out_channels, edge_dim))

        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_layers - 1)
        ])

        # self.dropout = nn.Dropout(p=0.1)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_weight)
        nn.init.zeros_(self.edge_bias)
        for layer in self.layers:
            layer.lin.reset_parameters()
            layer.edge_transform.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # generate edge feature
        edge_features = torch.matmul(edge_attr, self.edge_weight) + self.edge_bias

        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index, edge_features)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.layer_norms[i](x)
                x = self.dropout(x)

        return x


class WeightNet(nn.Module):
    def __init__(self, l: int, kernel_channels: list[int]):
        super(WeightNet, self).__init__()

        self.l = l
        self.kernel_channels = kernel_channels

        # Lists to store weight matrices and biases for each MLP layer
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()

        # Initialize weight and bias parameters for each layer
        for i, channels in enumerate(kernel_channels):
            if i == 0:
                # First layer: input dimension is 3 (source pos) + 3 (target pos) + 1 (distance) = 7
                self.Ws.append(torch.nn.Parameter(torch.empty(l, 3 + 3 + 1, channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))
            else:
                # Later layers: input dimension is previous hidden dimension
                self.Ws.append(torch.nn.Parameter(torch.empty(l, kernel_channels[i-1], channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))
        self.relu = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        # Initialize parameters with Kaiming uniform distribution and zero biases
        for i, channels in enumerate(self.kernel_channels):
            if i == 0:
                kaiming_uniform(self.Ws[0].data, size=[self.l, 3 + 3 + 1, channels])
            else:
                kaiming_uniform(self.Ws[i].data, size=[self.l, self.kernel_channels[i-1], channels])
            self.bs[i].data.fill_(0.0)

    def forward(self, input, idx):
        for i in range(len(self.kernel_channels)):
            W = torch.index_select(self.Ws[i], 0, idx)
            b = torch.index_select(self.bs[i], 0, idx)
            if i == 0:
                weight = self.relu(torch.bmm(input.unsqueeze(1), W).squeeze(1) + b)
            else:
                weight = self.relu(torch.bmm(weight.unsqueeze(1), W).squeeze(1) + b)
        # Final output: [B, kernel_channels[-1]]
        return weight

