from typing import List
import torch
from torch_geometric.nn import MLP, fps, global_max_pool, global_mean_pool, radius
from model.mmpgnet.modules_geometric import *
from model.mmpgnet.modules_similarity import *
from model.mmpgnet.modules_energy import *
from model.mmpgnet.EdgeGCN import EdgeGCN, WeightNet
from utils import set_seed
set_seed()

NUM_EXPERTS = 10
NUM_EXPERTS_K = 6
DROPOUT = 0.1
MODEL_EDGEGCN_DROPOUT_P = 0.1
print("model.py EdgeGCN dropout_p={}".format(MODEL_EDGEGCN_DROPOUT_P))


class MultiViewMoE(nn.Module):
    """
    Forward pass of the Multi-View Mixture-of-Experts (MoE) network.
    This method performs expert routing separately for each view based on global graph features,
    computes expert outputs for each node, and aggregates them using sparse top-k gating.
    """
    def __init__(self, input_size=256, output_size=256, hidden_size= 256, num_experts=8,
                 num_views=3, edge_dim=24, k=4, coef=1e-2):
        super(MultiViewMoE, self).__init__()
        self.num_experts = num_experts
        self.num_views = num_views
        self.k = k
        self.loss_coef = coef

        # Each expert is an EdgeGCN
        self.experts = nn.ModuleList([
            EdgeGCN(
                in_channels=input_size,
                out_channels=output_size,
                num_layers=1,
                hidden_channels=hidden_size,
                edge_dim=edge_dim,
                dropout=MODEL_EDGEGCN_DROPOUT_P,
            ) for _ in range(num_experts)
        ])

        # Gating network that maps global view features to expert scores
        self.gating_body = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
        )

        # Clean logits for gating
        self.w_gate = nn.Linear(2048, num_experts)
        # Noise scale generator
        self.w_noise = nn.Linear(input_size, num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def cv_squared(self, x):
        """Coefficient of variation squared: used for load balancing loss."""
        eps = 1e-10
        if x.numel() <= 1:
            return torch.zeros(1, device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
                Compute top-k expert routing with optional noise injection (for training).
                Returns gate values, selected indices, and auxiliary values for computing load balancing.
        """
        clean_logits = self.w_gate(x)
        if train:
            raw_std = self.w_noise(x)
            noise_std = self.softplus(raw_std) + noise_epsilon
            logits = clean_logits + torch.randn_like(clean_logits) * noise_std
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(self.k, dim=1)
        top_k_gates = self.softmax(top_logits)

        # Compute load and importance for auxiliary loss
        zeros = torch.zeros_like(logits)
        gates_for_loss = zeros.scatter(1, top_indices, 1)
        load = gates_for_loss.sum(dim=0)

        return top_k_gates, top_indices, gates_for_loss, load, logits

    def forward(self, x_views, edge_index_views, batch, edge_attr_views, istrain):
        """
         Forward pass of the multi-view MoE network.
         For each view, this method computes view-specific expert routing using global features,
         applies expert networks in parallel, and aggregates outputs via sparse top-k gating.
         """
        top_k_gates_list = []
        top_indices_list = []
        gates_for_loss_list = []
        load_list = []

        # Global features before gating (for auxiliary tasks)
        subclassification = []
        # Norm-based confidence scores
        subclassification_weights = []
        # --- Phase 1: View-wise routing and gate computation ---
        for v in range(self.num_views):
            # 1. Compute global pooled features from current view
            gf = global_mean_pool(x_views[v], batch)
            gfh = self.gating_body(gf)

            subclassification.append(gfh)

            # 2. Apply view-specific gating
            top_k_gates, top_indices, gates_for_loss, load, logits = self.noisy_top_k_gating(gfh, istrain)

            # 3. Store gating info for later use
            top_k_gates_list.append(top_k_gates)
            top_indices_list.append(top_indices)
            gates_for_loss_list.append(gates_for_loss)
            load_list.append(load)

            # 4. Compute normalized confidence scores (per view, per graph)
            norms = logits.norm(p=2, dim=1)
            eps = 1e-6
            min_val = norms.min()
            max_val = norms.max()
            weights = (norms - min_val) / (max_val - min_val + eps) + 0.1
            subclassification_weights.append(weights)

        # --- Phase 2: Load balancing loss ---
        all_gates_for_loss = torch.cat(gates_for_loss_list, dim=0)
        importance = all_gates_for_loss.sum(dim=0)

        all_load = torch.stack(load_list).sum(dim=0)

        # 计算总的负载均衡损失
        loss = (self.cv_squared(importance) + self.cv_squared(all_load)) * self.loss_coef

        # --- Phase 3: Expert computation and sparse aggregation ---
        final_x_nodes = []
        for v in range(self.num_views):
            x_v = x_views[v]  # [N, F]
            edge_index_v = edge_index_views[v]
            edge_attr_v = edge_attr_views[v]

            # 1) Compute outputs from all experts in parallel: [N, E, F]
            expert_outs = torch.stack([
                self.experts[i](x_v, edge_index_v, edge_attr_v)
                for i in range(self.num_experts)
            ], dim=1)  # shape [N, E, F]

            # 2) Get top-k expert indices and gate values per node
            node_top_indices = top_indices_list[v][batch]  # [N, k]
            node_top_values = top_k_gates_list[v][batch]  # [N, k]
            # 3) Build sparse gate tensor [N, E] using scatter
            gates = torch.zeros(
                x_v.size(0),
                self.num_experts,
                device=x_v.device,
                dtype=node_top_values.dtype
            )
            gates.scatter_(1, node_top_indices, node_top_values)
            # 4) Weighted sum across experts to get final node features [N, F]
            view_output = (expert_outs * gates.unsqueeze(-1)).sum(dim=1)

            final_x_nodes.append(view_output)
        # --- Phase 4: Aggregate view-level graph embeddings ---
        graph_outs = [global_mean_pool(xn, batch) for xn in final_x_nodes]
        x_graph = torch.stack(graph_outs, dim=0)
        return final_x_nodes, x_graph, loss, subclassification, subclassification_weights

class Linear(nn.Module):
    # A customizable linear projection block with optional BatchNorm, LeakyReLU activation, and Dropout.
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        return self.module(x)

class MLP(nn.Module):
    # A flexible two-layer MLP block with optional batch normalization, dropout, and LeakyReLU activation.
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        # Optional BatchNorm on input
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        # First activation and dropout
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        # First Linear layer: either directly to output or to hidden layer
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        # Optional BatchNorm after first Linear
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        # Second activation
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)
    def forward(self, input):
        # Tensor: Output tensor of shape [B, out_channels]
        return self.module(input)

# chemical-functional similarity-aware graph construction with residual graph block
class BasicBlock_similarity(nn.Module):
    def __init__(self,
                 l: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock_similarity, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = MMPG_similarity(l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x, ori, pos, seq, batch, edge_index, kernel_weight = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out, ori, pos, seq, batch, edge_index, kernel_weight

# geometric-aware graph construction with residual graph block
class BasicBlock_geometric(nn.Module):
    def __init__(self,
                 r: float,
                 l: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock_geometric, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)

        self.conv = MMPG_geometric(r =r, l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)

    def forward(self, x, pos, seq, ori, batch):
        identity = self.identity(x)
        x = self.input(x)
        x, ori, pos, seq, batch, edge_index, kernel_weight = self.conv(x, pos, seq, ori, batch)
        out = self.output(x) + identity
        return out, ori, pos, seq, batch, edge_index, kernel_weight

# physical-energetic aware graph construction with residual graph block
class BasicBlock_energy(nn.Module):
    def __init__(self,
                 l: float,
                 kernel_channels: list[int],
                 in_channels: int,
                 out_channels: int,
                 base_width: float = 16.0,
                 batch_norm: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:

        super(BasicBlock_energy, self).__init__()

        if in_channels != out_channels:
            self.identity = Linear(in_channels=in_channels,
                                  out_channels=out_channels,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope,
                                  momentum=momentum)
        else:
            self.identity = nn.Sequential()

        width = int(out_channels * (base_width / 64.))
        self.input = MLP(in_channels=in_channels,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope,
                         momentum=momentum)
        self.conv = MMPG_energy(l=l, kernel_channels=kernel_channels, in_channels=width, out_channels=width)
        self.output = Linear(in_channels=width,
                             out_channels=out_channels,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope,
                             momentum=momentum)
    def forward(self, x, pos, seq, ori, batch, name, edge_index):
        identity = self.identity(x)
        x = self.input(x)
        x, ori, pos, seq, batch,  edge_index, kernel_weight = self.conv(x, pos, seq, ori, batch, name, edge_index)
        out = self.output(x) + identity
        return out, ori, pos, seq, batch, edge_index, kernel_weight

class Model(nn.Module):
    """
    Main PyTorch model that constructs and processes protein graphs from three distinct
    perspectives (Similarity, Geometric, Energy) and fuses them using a
    Mixture-of-Experts (MoE) module.
    """
    def __init__(self,
                 geometric_radii: List[float],
                 sequential_kernel_size: float,
                 kernel_channels: List[int],
                 channels: List[int],
                 base_width: float = 16.0,
                 embedding_dim: int = 16,
                 batch_norm: bool = True,
                 dropout: float = DROPOUT,
                 bias: bool = False,
                 num_classes: int = 384) -> nn.Module:

        """
        Initializes the model architecture, including embeddings, parallel backbones,
        the MoE module, and final classifiers.
        """

        super().__init__()
        print(f"===== dropout={dropout}")

        assert (len(geometric_radii) == len(channels)), "Model: 'geometric_radii' and 'channels' should have the same number of elements!"

        # --- 1. Initial Feature Embedders ---
        # Separate embeddings for each of the three perspectives to start with different features.
        self.embedding_s = torch.nn.Embedding(num_embeddings=26, embedding_dim=embedding_dim)
        self.embedding_d = torch.nn.Embedding(num_embeddings=26, embedding_dim=embedding_dim)
        self.embedding_e = torch.nn.Embedding(num_embeddings=26, embedding_dim=embedding_dim)
        self.sidechainemb = torch.nn.Linear(8, 16)
        self.relativecoord = torch.nn.Linear(9, 32)
        self.local_mean_pool_geometric = AvgPooling_geometric()
        self.local_mean_pool_similarity = AvgPooling_similarity()
        self.local_mean_pool_energy = AvgPooling_energy()

        layers_similarity = []
        layers_geometric = []
        layers_energy = []
        in_channels_s = 32
        in_channels_d = 32
        in_channels_e = 32
        channels_s = channels_d =channels_e = channels

        # Each of the three branches (similarity, energy, geometric).
        for i in range(len(channels)):
            layers_similarity.append(BasicBlock_similarity(
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels_s,
                                     out_channels = channels_s[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))

            layers_similarity.append(BasicBlock_similarity(
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels_s[i],
                                     out_channels = channels_s[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))

            layers_energy.append(BasicBlock_energy(
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels_e,
                                     out_channels = channels_e[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))

            layers_energy.append(BasicBlock_energy(
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels_e[i],
                                     out_channels = channels_e[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))

            layers_geometric.append(BasicBlock_geometric(r = geometric_radii[i],
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = in_channels_d,
                                     out_channels = channels_d[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))

            layers_geometric.append(BasicBlock_geometric(r = geometric_radii[i],
                                     l = sequential_kernel_size,
                                     kernel_channels = kernel_channels,
                                     in_channels = channels_d[i],
                                     out_channels = channels_d[i],
                                     base_width = base_width,
                                     batch_norm = batch_norm,
                                     dropout = dropout,
                                     bias = bias))

            in_channels_s = channels_s[i]
            in_channels_d = channels_d[i]
            in_channels_e = channels_e[i]

        self.layers_similarity = nn.Sequential(*layers_similarity)
        self.layers_geometric = nn.Sequential(*layers_geometric)
        self.layers_energy = nn.Sequential(*layers_energy)

        self.classifier = MLP(in_channels=channels[-1]*3,
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

        self.classifier_o1 = MLP(in_channels=channels[-1]*3,
                              mid_channels=max(channels[-1], num_classes),
                              out_channels=num_classes,
                              batch_norm=batch_norm,
                              dropout=dropout)

        # Initialize the Multi-View Mixture-of-Experts module with 3 views and sparse top-k expert routing over 2048-dim features.
        self.moe = MultiViewMoE(
            input_size=2048,
            output_size=2048,
            hidden_size=2048,
            num_experts=NUM_EXPERTS,
            num_views=3,
            edge_dim=24,
            k=NUM_EXPERTS_K
        )
        print(f"====== num_experts={NUM_EXPERTS}, k={NUM_EXPERTS_K} ======")

    def forward(self, x, pos, seq, ori, batch, sidechain, name, istrain):

        # Forward pass through the similarity, geometric, and energy branches.
        x_s = torch.cat((self.embedding_s(x), self.sidechainemb(sidechain)), dim=1)
        x_d = torch.cat((self.embedding_d(x), self.sidechainemb(sidechain)), dim=1)
        x_e = torch.cat((self.embedding_e(x), self.sidechainemb(sidechain)), dim=1)

        # Apply stacked similarity blocks
        for i, layer in enumerate(self.layers_similarity):
            if i == 0:
                x_s, ori_s, pos_s, seq_s, batch_s, edge_index_s, kernel_weight_s = layer(x_s, pos, seq, ori, batch)
            else:
                x_s, ori_s, pos_s, seq_s, batch_s, edge_index_s, kernel_weight_s = layer(x_s, pos_s, seq_s, ori_s, batch_s)
            if i == 7:
                a = 1
            elif i % 2 == 1:
                x_s, pos_s, seq_s, ori_s, batch_s, = self.local_mean_pool_similarity(x_s, pos_s, seq_s, ori_s, batch_s)

        # Apply stacked geometric blocks
        for i, layer in enumerate(self.layers_geometric):
            if i == 0:
                x_d, ori_d, pos_d, seq_d, batch_d, edge_index_d, kernel_weight_d = layer(x_d, pos, seq, ori, batch)
            else:
                x_d, ori_d, pos_d, seq_d, batch_d, edge_index_d, kernel_weight_d = layer(x_d, pos_d, seq_d, ori_d, batch_d)
            if i == 7:
                a = 1
            elif i % 2 == 1:
                x_d, pos_d, seq_d, ori_d, batch_d = self.local_mean_pool_geometric(x_d, pos_d, seq_d, ori_d, batch_d)

        # Apply stacked energetic blocks
        for i, layer in enumerate(self.layers_energy):
            if i == 0:
                x_e, ori_e, pos_e, seq_e, batch_e,  edge_index_e, kernel_weight_e = layer(x_e, pos, seq, ori, batch, name, None)
            else:
                x_e, ori_e, pos_e, seq_e, batch_e, edge_index_e, kernel_weight_e = layer(x_e, pos_e, seq_e, ori_e, batch_e, name, edge_index_e)
            if i == 7:
                a = 1
            elif i % 2 == 1:
                x_e, pos_e, seq_e, ori_e, batch_e,  edge_index_e = self.local_mean_pool_energy(x_e, pos_e, seq_e, ori_e, batch_e, edge_index_e)

        x_views = [x_s, x_d, x_e]
        edge_index_views = [edge_index_s, edge_index_d, edge_index_e]
        edge_attr_views = [kernel_weight_s,kernel_weight_d,kernel_weight_e]

        # Apply moe module
        x_nodes, x_moe_graph, moe_loss, subclassification, subclassification_weights = self.moe(
            x_views, edge_index_views, batch_s, edge_attr_views, istrain
        )
        fusionsemantics = torch.concat((x_moe_graph[0],x_moe_graph[1],x_moe_graph[2]),dim=1)

        out_allviews = self.classifier(fusionsemantics)
        out_view1 = self.classifier_o1(torch.cat((subclassification[0],subclassification[1],subclassification[2]),dim=1))

        return out_allviews, out_view1, moe_loss
