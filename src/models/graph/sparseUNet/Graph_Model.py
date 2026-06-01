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

    def aggregate_nodes(self, point_features, labels):
        """
        """

        node_mean_base = scatter_mean(point_features, labels, dim=0)
        node_max_base = scatter_max(point_features, labels, dim=0)[0]

        rel_feat_mean = node_mean_base[labels] - point_features
        attn_mean = scatter_softmax(self.pooling_attn_mean(rel_feat_mean), labels, dim=0)
        feat_mean = scatter_sum(attn_mean * point_features, labels, dim=0)
        

        rel_feat_max = node_max_base[labels] - point_features
        attn_max = scatter_softmax(self.pooling_attn_max(rel_feat_max), labels, dim=0)
        feat_max = scatter_sum(attn_max * point_features, labels, dim=0)

        return self.feature_fusion(torch.cat([feat_mean, feat_max], dim=-1))

    def forward(self, voxels, indices, batch_size, spatial_shape, point_labels, edge_index, node_gt_ids=None):
        """
        Args:
            indices: [N, 4] (batch_idx, z, y, x)
        """

        coords = indices[:, 1:].float()
        spatial = torch.tensor(spatial_shape, device=coords.device).float()
        voxels_in = torch.cat([voxels, coords / spatial], dim=1)

        input_tensor = spconv.SparseConvTensor(voxels_in, indices, spatial_shape, batch_size)
        x = self.input_conv(input_tensor)
        unet_out = self.sparse_unet(x)
        point_features = self.output_layer(unet_out.features) # [N_points, 32]

        node_features = self.aggregate_nodes(point_features, point_labels) # [N_frags, gcn_hidden]

        gt_prototypes = None
        if node_gt_ids is not None:

            point_to_gt_raw = node_gt_ids[point_labels] 
            

            unique_gts, point_to_indices = torch.unique(point_to_gt_raw, return_inverse=True)
            

            gt_prototypes = self.aggregate_nodes(point_features, point_to_indices)

        h = self.gat1(node_features, edge_index)
        h = F.elu(self.gat_norm1(h)) + node_features
        h_final = self.gat2(h, edge_index)
        h_final = F.elu(self.gat_norm2(h_final)) + h

        u_idx, v_idx = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h_final[u_idx], h_final[v_idx]], dim=-1)
        action_logits = self.policy_head(edge_repr)

        return action_logits, node_features, gt_prototypes