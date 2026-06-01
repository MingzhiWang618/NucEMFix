import torch
import h5py
import json
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class PointNetNucleiDataset(Dataset):
    def __init__(self, h5_path, dist_threshold=7.0, target_n=15000, normalize=True):
        self.h5_path = h5_path
        self.dist_threshold = dist_threshold
        self.target_n = target_n
        self.normalize = normalize
        
        with h5py.File(self.h5_path, 'r') as f:

            all_ids = list(f['sampled_coords'].keys())
            self.sample_ids = []
            for sid in all_ids:
                if f['sampled_coords'][sid].shape[0] > 128:
                    self.sample_ids.append(sid)

    def __len__(self):
        return len(self.sample_ids)

    def _normalize_points(self, coords):
        """
        """
        centroid = np.mean(coords, axis=0)
        coords = coords - centroid
        m = np.max(np.sqrt(np.sum(coords**2, axis=1)))
        if m > 0:
            coords = coords / m
        return coords.astype(np.float32)

    @torch.no_grad()
    def _get_geometry_constrained_graph(self, coords, labels, mapping, id_map, unique_ids, neg_pos_ratio=50.0):
        """
        """
        obj_pts_dict = {uid: torch.from_numpy(coords[labels == uid]).float() for uid in unique_ids}
        obj_bboxes = {uid: np.concatenate([coords[labels == uid].min(axis=0), 
                                          coords[labels == uid].max(axis=0)]) for uid in unique_ids}

        pos_edges, neg_edges = [], []
        uids_list = list(unique_ids)
        
        for i in range(len(uids_list)):
            for j in range(i + 1, len(uids_list)):
                id1, id2 = uids_list[i], uids_list[j]
                

                b1, b2 = obj_bboxes[id1], obj_bboxes[id2]
                dist_sq = np.sum(np.maximum(0, np.maximum(b1[:3] - b2[3:], b2[:3] - b1[3:]))**2)
                if dist_sq > self.dist_threshold**2: continue

                p1, p2 = obj_pts_dict[id1], obj_pts_dict[id2]
                s1 = p1[torch.randint(0, len(p1), (min(len(p1), 200),))]
                s2 = p2[torch.randint(0, len(p2), (min(len(p2), 200),))]
                dists = torch.cdist(s1.unsqueeze(0), s2.unsqueeze(0)).squeeze(0)
                
                if torch.any(dists <= self.dist_threshold):
                    u_idx, v_idx = id_map[id1], id_map[id2]

                    is_same = 1.0 if mapping.get(id1, -1) == mapping.get(id2, -2) and mapping.get(id1, -1) != -1 else 0.0
                    
                    if is_same == 1.0:
                        pos_edges.append(([u_idx, v_idx], 1.0))
                    else:
                        neg_edges.append(([u_idx, v_idx], 0.0))

        if len(pos_edges) > 0:
            max_neg = int(len(pos_edges) * neg_pos_ratio)
            if len(neg_edges) > max_neg:
                neg_edges = random.sample(neg_edges, max_neg)
        else:
            neg_edges = random.sample(neg_edges, min(len(neg_edges), 50))

        final_edges = pos_edges + neg_edges
        random.shuffle(final_edges)

        if not final_edges:
            return np.zeros((2, 0)), np.zeros(0)

        edge_index = np.array([e[0] for e in final_edges]).T
        edge_gt = np.array([e[1] for e in final_edges])
        return edge_index, edge_gt

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        with h5py.File(self.h5_path, 'r') as f:
            raw_coords = f['sampled_coords'][sample_id][:].astype(np.float32)
            raw_labels = f['sampled_labels'][sample_id][:].astype(np.int32)
            mapping = json.loads(f['mappings'][sample_id][()])
            mapping = {int(k): int(v) for k, v in mapping.items()}

        n_pts = raw_coords.shape[0]
        choice = np.random.choice(n_pts, self.target_n, replace=(n_pts < self.target_n))
        coords = raw_coords[choice]
        labels = raw_labels[choice]

        unique_ids = np.unique(labels)
        unique_ids = unique_ids[unique_ids > 0]
        id_map = {old_id: i for i, old_id in enumerate(unique_ids)}
        

        node_gt_ids = np.array([mapping.get(uid, -1) for uid in unique_ids], dtype=np.int32)
        

        point_to_node_mask = np.array([id_map[l] if l in id_map else -1 for l in labels])
        

        edge_index, edge_gt = self._get_geometry_constrained_graph(coords, labels, mapping, id_map, unique_ids)

        model_input_coords = self._normalize_points(coords)

        return {
            "points": torch.from_numpy(model_input_coords),   # [N, 3]
            "point_labels": torch.from_numpy(point_to_node_mask).long(), # [N]
            "edge_index": torch.from_numpy(edge_index).long(), 
            "edge_gt": torch.from_numpy(edge_gt).float(),
            "node_gt_ids": torch.from_numpy(node_gt_ids).long(),
            "num_nodes": len(unique_ids)
        }

def pointnet_nuclei_collate_fn(batch):

    points = torch.stack([item['points'] for item in batch], dim=0)
    

    all_point_labels = []
    all_edge_index = []
    all_edge_gt = []
    all_node_gt_ids = []
    
    node_offset = 0
    for item in batch:

        p_labels = item['point_labels'].clone()
        mask = (p_labels != -1)
        p_labels[mask] += node_offset
        all_point_labels.append(p_labels)
        

        if item['edge_index'].shape[1] > 0:
            all_edge_index.append(item['edge_index'] + node_offset)
            all_edge_gt.append(item['edge_gt'])
            
        all_node_gt_ids.append(item['node_gt_ids'])
        
        node_offset += item['num_nodes']

    return {
        "points": points.contiguous(),
        "point_labels": torch.cat(all_point_labels, dim=0),
        "edge_index": torch.cat(all_edge_index, dim=1) if all_edge_index else None,
        "edge_gt": torch.cat(all_edge_gt, dim=0) if all_edge_gt else None,
        "node_gt_ids": torch.cat(all_node_gt_ids, dim=0), # [Total_Nodes]
        "batch_size": len(batch)
    }