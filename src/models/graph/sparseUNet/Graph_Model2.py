import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_softmax

class SubNucleiAggregationNet(nn.Module):
    def __init__(self, unet_planes, gcn_hidden, media=32, n_heads=4):
        super().__init__()
        

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(4, unet_planes[0], kernel_size=3, padding=1, bias=False, indice_key="subm0"),
        )

        from .Sparse_UNet import UBlock
        self.sparse_unet = UBlock(nPlanes=unet_planes)
        

        self.output_layer = nn.Sequential(
            nn.BatchNorm1d(unet_planes[0]),
            nn.ReLU(inplace=True)
        )

        self.pooling_attn_mean = nn.Sequential(
            nn.Linear(unet_planes[0], media),
            nn.BatchNorm1d(media),
            nn.ReLU(),
            nn.Linear(media, 1)
        )
        self.pooling_attn_max = nn.Sequential(
            nn.Linear(unet_planes[0], media),
            nn.BatchNorm1d(media),
            nn.ReLU(),
            nn.Linear(media, 1)
        )
        

        self.feature_fusion = nn.Sequential(
            nn.Linear(2 * unet_planes[0], gcn_hidden),
            nn.BatchNorm1d(gcn_hidden),
            nn.ReLU(),
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.BatchNorm1d(gcn_hidden)
        )

        self.gat1 = GATv2Conv(gcn_hidden, gcn_hidden // n_heads, heads=n_heads, dropout=0.1)
        self.gat_norm1 = nn.LayerNorm(gcn_hidden)
        

        self.gat2 = GATv2Conv(gcn_hidden, gcn_hidden, heads=1, concat=False, dropout=0.1)
        self.gat_norm2 = nn.LayerNorm(gcn_hidden)
        

        self.policy_head = nn.Sequential(
            nn.Linear(gcn_hidden * 2, gcn_hidden),
            nn.ReLU(),
            nn.Linear(gcn_hidden, 1)
        )

    def forward(self, voxels, indices, batch_size, spatial_shape, point_labels, edge_index):

        coords = indices[:, 1:].float()      # [N, 3]  (z,y,x)
        spatial = torch.tensor(spatial_shape, device=coords.device).float()
        coords_norm = coords / spatial       # normalize to [0,1]

        voxels = torch.cat([voxels, coords_norm], dim=1)   # [N, 4]

        # A. Sparse Convolution
        input_tensor = spconv.SparseConvTensor(voxels, indices, spatial_shape, batch_size)
        x = self.input_conv(input_tensor)
        unet_out = self.sparse_unet(x)
        point_features = self.output_layer(unet_out.features)

        node_mean_base = scatter_mean(point_features, point_labels, dim=0)
        node_max_base = scatter_max(point_features, point_labels, dim=0)[0]

        rel_feat_mean = node_mean_base[point_labels] - point_features
        attn_mean = scatter_softmax(self.pooling_attn_mean(rel_feat_mean), point_labels, dim=0)
        feat_mean = scatter_sum(attn_mean * point_features, point_labels, dim=0)
        
        rel_feat_max = node_max_base[point_labels] - point_features
        attn_max = scatter_softmax(self.pooling_attn_max(rel_feat_max), point_labels, dim=0)
        feat_max = scatter_sum(attn_max * point_features, point_labels, dim=0)

        node_features = self.feature_fusion(torch.cat([feat_mean, feat_max], dim=-1))

        h = self.gat1(node_features, edge_index)
        h = F.elu(self.gat_norm1(h)) + node_features
        

        h_final = self.gat2(h, edge_index)
        h_final = F.elu(self.gat_norm2(h_final)) + h

        # D. Policy Head
        u_idx, v_idx = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h_final[u_idx], h_final[v_idx]], dim=-1)
        action_logits = self.policy_head(edge_repr)
        #return action_logits
        return action_logits, h_final

import os
import sys
import warnings
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=FutureWarning)

def nuclei_collate_fn(batch):
    voxels_list = []
    indices_list = []
    point_labels_list = []
    edge_index_list = []
    edge_gt_list = []
    
    node_offset = 0

    max_spatial = np.max([d['spatial_shape'] for d in batch], axis=0)
    max_spatial = ((max_spatial // 32 + 1) * 32).tolist()
    
    for i, data in enumerate(batch):

        b_idx = torch.full((data['indices'].shape[0], 1), i, dtype=torch.int)
        indices_list.append(torch.cat([b_idx, data['indices']], dim=1))
        voxels_list.append(data['voxels'])
        

        point_labels_list.append(data['point_labels'] + node_offset)
        
        if data['edge_index'].shape[1] > 0:
            edge_index_list.append(data['edge_index'] + node_offset)
            edge_gt_list.append(data['edge_gt'])
            

        num_nodes = data['point_labels'].max().item() + 1
        node_offset += num_nodes

    return {
        "voxels": torch.cat(voxels_list, dim=0),
        "indices": torch.cat(indices_list, dim=0),
        "point_labels": torch.cat(point_labels_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2,0), dtype=torch.long),
        "edge_gt": torch.cat(edge_gt_list, dim=0) if edge_gt_list else torch.zeros(0),
        "spatial_shape": max_spatial,
        "batch_size": len(batch)
    }

def train_one_step(model, batch, optimizer, device):
    optimizer.zero_grad()
    

    voxels = batch['voxels'].to(device)
    indices = batch['indices'].to(device)
    point_labels = batch['point_labels'].to(device)
    edge_index = batch['edge_index'].to(device)
    edge_gt = batch['edge_gt'].to(device)
    

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        logits = model(
            voxels, indices, batch['batch_size'], 
            batch['spatial_shape'], point_labels, edge_index
        )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    if logits.shape[0] > 0:
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), edge_gt)
        loss.backward()
        optimizer.step()
        return loss.item()
    return 0.0

if __name__ == "__main__":
    import argparse
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from src.datasets.graph.dataset import NucleiDistanceDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, required=True, help="Path to HDF5 dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NucleiDistanceDataset(args.h5_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=nuclei_collate_fn)
    model = SubNucleiAggregationNet(unet_planes=[32, 64, 128], gcn_hidden=128).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for batch in loader:
        loss_val = train_one_step(model, batch, optimizer, DEVICE)
        print(f"Loss: {loss_val:.4f}")
        break