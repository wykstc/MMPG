import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_max, scatter_min, scatter_mean, scatter_sum
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from model.mmpgnet.utils import get_adj_tensor, get_normalize_adj_tensor, to_dense_adj, dense_to_sparse, switch_edge, drop_feature, kaiming_uniform
from model.mmpgnet.EdgeGCN import EdgeGCN, WeightNet
import pickle
from torch_geometric.data import Data, Batch
from utils import set_seed
set_seed()

# Select energy-based connection. There are four different choices.
ENERGY_PATH = "/FOLD_korp_abs_m0.5.pkl"
# ENERGY_PATH = "FOLD_korp_abs_m1.0.pkl"
# ENERGY_PATH = "FOLD_korp_abs_m1.5.pkl"
# ENERGY_PATH = "FOLD_korp_abs_m2.0.pkl"
print(f"energy path is {ENERGY_PATH}")
energy = open(ENERGY_PATH, 'rb')
graphconstruct = pickle.load(energy)


class MMPG_energy(MessagePassing):
    def __init__(self,  l: float, kernel_channels: list[int], in_channels: int, out_channels: int, add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'sum')
        super().__init__(**kwargs)
        self.l = l
        self.kernel_channels = kernel_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.WeightNet = WeightNet(l, kernel_channels)
        self.W = torch.nn.Parameter(torch.empty(kernel_channels[-1] * in_channels, out_channels))

        self.add_self_loops = add_self_loops

        self.reset_parameters()
        self.pointfeature = None


        # Define multiple EdgeGCN layers with increasing feature dimensions
        self.gcn1 = EdgeGCN(
            in_channels=256,
            out_channels=256,
            num_layers=3,
            hidden_channels=256,
            edge_dim=24
        )

        self.gcn2 = EdgeGCN(
            in_channels=512,
            out_channels=512,
            num_layers=3,
            hidden_channels=512,
            edge_dim=24
        )

        self.gcn3 = EdgeGCN(
            in_channels=1024,
            out_channels=1024,
            num_layers=3,
            hidden_channels=1024,
            edge_dim=24
        )

        self.gcn4 = EdgeGCN(
            in_channels=2048,
            out_channels=2048,
            num_layers=3,
            hidden_channels=2048,
            edge_dim=24
        )
        self.update_data_ls = []

        self.pos_dict1 = {}
        self.pos_dict2 = {}
        self.pos_dict3 = {}
        self.pos_dict4 = {}

    def reset_parameters(self):
        self.WeightNet.reset_parameters()
        kaiming_uniform(self.W.data, size=[self.kernel_channels * self.in_channels, self.out_channels])

    def build_data_list(self, graphconstruct, names):
        data_list = []
        for name in names:
            row, col = graphconstruct[name]
            edge_index = torch.tensor([row, col], dtype=torch.long)

            data = Data(edge_index=edge_index)
            data_list.append(data)
        return data_list

    def forward(self, x: OptTensor, pos: Tensor, seq: Tensor, ori: Tensor, batch: Tensor, name, edge_index=None) -> Tensor:
        # Construct graph data for the initial layer
        if x.shape[1] == 256:
            data_list = self.build_data_list(graphconstruct, name)
            batch1 = Batch.from_data_list(data_list)
            edge_index = batch1.edge_index
            edge_index = edge_index.to("cuda")

        # Compute edge features - Includes distance, orientation, sequence features
        edge_features, seq_idx, kernel_weight = self.compute_edge_features(edge_index,pos,ori,seq,)
        if x.shape[1] == 256:
            out = self.gcn1(x, edge_index, edge_attr=kernel_weight)
        elif x.shape[1] == 512:
            out = self.gcn2(x, edge_index, edge_attr=kernel_weight)
        elif x.shape[1] == 1024:
            out = self.gcn3(x, edge_index, edge_attr=kernel_weight)
        elif x.shape[1] == 2048:
            out = self.gcn4(x, edge_index, edge_attr=kernel_weight)

        return out, ori, pos, seq, batch, edge_index, kernel_weight

    def compute_edge_features(self, edge_index, pos, ori, seq):
        """
        Compute raw edge features for each edge without aggregation.
        """
        row, col = edge_index
        # Extract node-wise attributes for each edge
        pos_i, pos_j = pos[row], pos[col]
        ori_i, ori_j = ori[row], ori[col]
        seq_i, seq_j = seq[row], seq[col]

        # ---- Position-related features ----
        pos_diff = pos_j - pos_i
        distance = torch.norm(input=pos_diff, p=2, dim=-1, keepdim=True)
        pos_diff /= (distance + 1e-9)
        # Transform relative position into local coordinate frame of source node
        pos_diff = torch.matmul(ori_i.reshape((-1, 3, 3)), pos_diff.unsqueeze(2)).squeeze(2)

        # Compute directional alignment between orientation matrices
        ori_feat = torch.sum(input=ori_i.reshape((-1, 3, 3)) * ori_j.reshape((-1, 3, 3)), dim=2, keepdim=False)

        # Sequence-related features
        seq_diff = seq_j - seq_i
        s = self.l // 2
        seq_diff = torch.clamp(input=seq_diff, min=-s, max=s)
        seq_idx = (seq_diff + s).squeeze(1).to(torch.int64)
        # Combine all edge features
        edge_features = torch.cat([pos_diff, ori_feat, distance], dim=1)
        kernel_weight = self.WeightNet(edge_features, seq_idx)

        return edge_features, seq_idx, kernel_weight

# Performs average pooling over nodes based on pairwise sequence proximity
class AvgPooling_energy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, pos, seq, ori, batch, edge_index=None):
        idx = torch.div(seq.squeeze(1), 2, rounding_mode='floor')
        idx = torch.cat([idx, idx[-1].view((1,))])

        idx = (idx[0:-1] != idx[1:]).to(torch.float32)
        idx = torch.cumsum(idx, dim=0) - idx
        idx = idx.to(torch.int64)

        # Pool node features within each cluster
        x = scatter_mean(src=x, index=idx, dim=0)
        pos = scatter_mean(src=pos, index=idx, dim=0)
        seq = scatter_max(src=torch.div(seq, 2, rounding_mode='floor'), index=idx, dim=0)[0]
        ori = scatter_mean(src=ori, index=idx, dim=0)
        ori = torch.nn.functional.normalize(ori, 2, -1)
        batch = scatter_max(src=batch, index=idx, dim=0)[0]

        # Process edge_index
        if edge_index is not None:
            edge_index = self.pool_edge_index(edge_index, idx)

        return x, pos, seq, ori, batch, edge_index

    def pool_edge_index(self, edge_index, cluster):
        #  Map original node indices to pooled node indices
        row, col = edge_index
        row_pooled = cluster[row]
        col_pooled = cluster[col]

        # edges that connect nodes merged into the same pooled node
        mask = row_pooled != col_pooled
        row_pooled = row_pooled[mask]
        col_pooled = col_pooled[mask]

        # Handle duplicate edges: multiple original edges may now point to the same pair of pooled nodes
        edge_index_pooled = torch.stack([row_pooled, col_pooled], dim=0)

        # Remove duplicate edges
        edge_index_pooled = torch.unique(edge_index_pooled, dim=1)

        return edge_index_pooled
