import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean, scatter_max, scatter_sum, scatter_softmax

class PointNetSubNucleiNet(nn.Module):
    def __init__(self, gcn_hidden=128, point_feat_dim=128, media=32, n_heads=4):
        super().__init__()
        

        from .pointNet2 import PointNet2Backbone 
        self.backbone = PointNet2Backbone(input_channels=0, out_channels=point_feat_dim)
        

        self.pooling_attn_mean = nn.Sequential(
            nn.Linear(point_feat_dim, media),
            nn.InstanceNorm1d(media),
            nn.ReLU(),
            nn.Linear(media, 1)
        )
        self.pooling_attn_max = nn.Sequential(
            nn.Linear(point_feat_dim, media),
            nn.InstanceNorm1d(media),
            nn.ReLU(),
            nn.Linear(media, 1)
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(2 * point_feat_dim, gcn_hidden),
            nn.InstanceNorm1d(gcn_hidden),
            nn.ReLU(),
            nn.Linear(gcn_hidden, gcn_hidden),
            nn.InstanceNorm1d(gcn_hidden)
        )

        # 3. Graph Reasoning
        self.gat1 = GATv2Conv(gcn_hidden, gcn_hidden // n_heads, heads=n_heads, dropout=0.1)
        self.gat_norm1 = nn.LayerNorm(gcn_hidden)
        
        self.gat2 = GATv2Conv(gcn_hidden, gcn_hidden, heads=1, concat=False, dropout=0.1)
        self.gat_norm2 = nn.LayerNorm(gcn_hidden)
        
        # 4. Policy Head
        self.policy_head = nn.Sequential(
            nn.Linear(gcn_hidden * 2, gcn_hidden),
            nn.ReLU(),
            nn.Linear(gcn_hidden, 1)
        )

    def aggregate_nodes(self, point_features, labels):

        point_features = point_features.contiguous()
        

        node_mean_base = scatter_mean(point_features, labels, dim=0)
        node_max_base = scatter_max(point_features, labels, dim=0)[0]

        rel_feat_mean = node_mean_base[labels] - point_features
        attn_mean = scatter_softmax(self.pooling_attn_mean(rel_feat_mean), labels, dim=0)
        feat_mean = scatter_sum(attn_mean * point_features, labels, dim=0)
        

        rel_feat_max = node_max_base[labels] - point_features
        attn_max = scatter_softmax(self.pooling_attn_max(rel_feat_max), labels, dim=0)
        feat_max = scatter_sum(attn_max * point_features, labels, dim=0)

        return self.feature_fusion(torch.cat([feat_mean, feat_max], dim=-1))

    def forward(self, points, point_labels, edge_index, node_gt_ids=None):
        """
        Args:
            points: [B, N, 3] 
        """

        points = points.contiguous().float()
        feat_raw = self.backbone(points)
        

        B, C, N = feat_raw.shape
        point_features = feat_raw.permute(0, 2, 1).reshape(-1, C).contiguous() # [B*N, C]

        valid_mask = (point_labels >= 0)

        if not valid_mask.any():
            return None, None, None
            
        node_features = self.aggregate_nodes(point_features[valid_mask], point_labels[valid_mask])

        gt_prototypes = None
        if node_gt_ids is not None:

            point_to_gt_raw = node_gt_ids[point_labels[valid_mask]]

            _, point_to_indices = torch.unique(point_to_gt_raw, return_inverse=True)
            gt_prototypes = self.aggregate_nodes(point_features[valid_mask], point_to_indices)

        if edge_index is not None and edge_index.shape[1] > 0:
            h = self.gat1(node_features, edge_index)
            h = F.elu(self.gat_norm1(h)) + node_features
            h_final = self.gat2(h, edge_index)
            h_final = F.elu(self.gat_norm2(h_final)) + h

            u_idx, v_idx = edge_index[0], edge_index[1]
            edge_repr = torch.cat([h_final[u_idx], h_final[v_idx]], dim=-1)
            action_logits = self.policy_head(edge_repr).squeeze(-1)
        else:
            action_logits = torch.empty(0, device=points.device)

        return action_logits, node_features, gt_prototypes

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    B, N, C_in = 4, 15000, 3
    num_nodes_per_sample = [10, 15, 8, 12]
    total_nodes = sum(num_nodes_per_sample)
    

    points = torch.randn(B, N, C_in).to(device)
    

    point_labels_list = []
    node_offset = 0
    for n_nodes in num_nodes_per_sample:

        local_labels = torch.randint(0, n_nodes, (N,))
        point_labels_list.append(local_labels + node_offset)
        node_offset += n_nodes
    point_labels = torch.cat(point_labels_list).to(device)
    

    edge_index_list = []
    curr_offset = 0
    for n_nodes in num_nodes_per_sample:

        u, v = torch.meshgrid(torch.arange(n_nodes), torch.arange(n_nodes), indexing='ij')
        mask = u != v
        edges = torch.stack([u[mask] + curr_offset, v[mask] + curr_offset])
        edge_index_list.append(edges)
        curr_offset += n_nodes
    edge_index = torch.cat(edge_index_list, dim=1).to(device)
    

    node_gt_ids = torch.randint(0, 100, (total_nodes,)).to(device)

    model = PointNetSubNucleiNet(gcn_hidden=128, point_feat_dim=128).to(device)
    model.eval()

  
    with torch.no_grad():
        action_logits, nodes, gt_protos = model(points, point_labels, edge_index, node_gt_ids)
        
    print("Forward Success!")
    print(f"Action Logits Shape: {action_logits.shape}") # [E]
    print(f"Node Features Shape: {nodes.shape}")         # [total_nodes, 128]
    print(f"GT Prototypes Shape: {gt_protos.shape}")     # [unique_gts, 128]

if __name__ == "__main__":
    test_model()