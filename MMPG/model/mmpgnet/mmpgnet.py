import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import normalize
from model.mmpgnet.models import Model
from utils import set_seed
set_seed()

class MMpgNet(nn.Module):

    def __init__(
            self,
    ):
        super(MMpgNet, self).__init__()

        # radii for constructing graph for geometric graphs
        # sequential_kernel_size: size of sequential convolution kernels
        # kernel_channels: number of channels in kernel weight networks
        # channels: list of feature dimensions for each layer
        # base_width: base width for convolutional layers
        # num_classes: output dimension (number of classes)
        self.MMpgNet = Model(geometric_radii=[2 * 4, 3*4, 4*4, 5*4],
                           sequential_kernel_size=5,
                           kernel_channels=[24], channels=[256,512,1024,2048], base_width=64,
                           num_classes=1195)

    def orientation(self,pos):
        # Compute backbone direction vectors (normalized)
        u = normalize(X=pos[1:, :] - pos[:-1, :], norm='l2', axis=1)
        u1 = u[1:, :]
        u2 = u[:-1, :]
        b = normalize(X=u2 - u1, norm='l2', axis=1)
        # Compute normal vector
        n = normalize(X=np.cross(u2, u1), norm='l2', axis=1)
        # Compute orthogonal vector
        o = normalize(X=np.cross(b, n), norm='l2', axis=1)
        # Stack the frame vectors to form orientation matrices
        ori = np.stack([b, n, o], axis=1)
        # Pad the first and last positions to maintain length consistency
        return np.concatenate([np.expand_dims(ori[0], 0), ori, np.expand_dims(ori[-1], 0)], axis=0)


    def forward(self, batch_data, istrain):
        # Extract features and coordinates from batch_data
        z, ca, n, c, batch = torch.squeeze(batch_data.x.long()), batch_data.coords_ca, batch_data.coords_n, batch_data.coords_c, batch_data.batch
        sidechain = batch_data['side_chain_embs']
        coords_ca = ca.cpu().numpy()
        ori_matrix = self.orientation(coords_ca)
        ori = torch.tensor(ori_matrix, dtype=torch.float32).to('cuda')

        unique_elements, inverse_indices = torch.unique(batch, return_inverse=True)
        _, counts = torch.unique_consecutive(inverse_indices, return_counts=True)
        seq = torch.cat([torch.arange(1, count + 1, device='cuda:0') for count in counts]).unsqueeze(1)
        # Forward pass through the main model
        out, out_1, moe_loss = self.MMpgNet(z, ca, seq, ori, batch, sidechain, batch_data["id"], istrain)

        return out, out_1, moe_loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
