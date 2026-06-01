import torch
import h5py
import json
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class NucleiDistanceDataset(Dataset):
    def __init__(self, h5_path, dist_threshold=3.0, max_pts_for_dist=50000):
        self.h5_path = h5_path
        self.dist_threshold = dist_threshold
        self.max_pts_for_dist = max_pts_for_dist
        with h5py.File(self.h5_path, 'r') as f:
            self.sample_ids = list(f['sampled_coords'].keys())

    def __len__(self):
        return len(self.sample_ids)

    @torch.no_grad()
    def _get_geometry_constrained_graph(self, coords, labels, mapping, id_map, unique_ids, neg_pos_ratio=25.0):
        """
        """
        obj_pts_dict = {}
        obj_bboxes = {}
        for uid in unique_ids:
            pts = coords[labels == uid]
            if len(pts) > 0:
                obj_pts_dict[uid] = torch.from_numpy(pts).float()
                obj_bboxes[uid] = np.concatenate([pts.min(axis=0), pts.max(axis=0)])

        pos_edges = []
        neg_edges = []
        uids_list = list(unique_ids)
        
        for i in range(len(uids_list)):
            for j in range(i + 1, len(uids_list)):
                id1, id2 = uids_list[i], uids_list[j]
                if id1 not in obj_pts_dict or id2 not in obj_pts_dict: continue
                

                b1, b2 = obj_bboxes[id1], obj_bboxes[id2]
                dz = max(0, b2[0] - b1[3], b1[0] - b2[3])
                dy = max(0, b2[1] - b1[4], b1[1] - b2[4])
                dx = max(0, b2[2] - b1[5], b1[2] - b2[5])
                if (dz**2 + dy**2 + dx**2) > self.dist_threshold**2:
                    continue

                p1, p2 = obj_pts_dict[id1], obj_pts_dict[id2]

                dists = torch.cdist(p1[::10].unsqueeze(0), p2[::10].unsqueeze(0), p=2).squeeze(0)
                
                if torch.any(dists <= self.dist_threshold):
                    u_idx, v_idx = id_map[id1], id_map[id2]
                    is_same = 1.0 if mapping.get(id1, -1) == mapping.get(id2, -2) else 0.0
                    
                    if is_same == 1.0:
                        pos_edges.append(([u_idx, v_idx], 1.0))
                    else:
                        neg_edges.append(([u_idx, v_idx], 0.0))

        num_pos = len(pos_edges)
        num_neg = len(neg_edges)

        if num_pos > 0:

            max_neg = int(num_pos * neg_pos_ratio)
            if num_neg > max_neg:
                neg_edges = random.sample(neg_edges, max_neg)
        else:

            neg_edges = random.sample(neg_edges, min(num_neg, 20))

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
            coords = f['sampled_coords'][sample_id][:]
            labels = f['sampled_labels'][sample_id][:]
            mapping = json.loads(f['mappings'][sample_id][()])
            mapping = {int(k): int(v) for k, v in mapping.items()}

        unique_ids = np.unique(labels)
        id_map = {old_id: i for i, old_id in enumerate(unique_ids)}
        point_labels_encoded = np.array([id_map[lid] for lid in labels])
        
        edge_index, edge_gt = self._get_geometry_constrained_graph(coords, labels, mapping, id_map, unique_ids)
        
        max_coords = coords.max(axis=0)
        spatial_shape = ((max_coords // 16 + 1) * 16).astype(int).tolist()
        
        return {
            "voxels": torch.ones((len(coords), 1), dtype=torch.float32),
            "indices": torch.from_numpy(coords).int(),
            "point_labels": torch.from_numpy(point_labels_encoded).long(),
            "edge_index": torch.from_numpy(edge_index).long(),
            "edge_gt": torch.from_numpy(edge_gt).float(),
            "spatial_shape": spatial_shape
        }

def nuclei_collate_fn(batch):
    voxels_list = []
    indices_list = []
    point_labels_list = []
    edge_index_list = []
    edge_gt_list = []
    node_gt_ids_list = []
    
    node_offset = 0
    nuc_id_offset = 0
    
    max_spatial = np.max([d['spatial_shape'] for d in batch], axis=0).tolist()
    
    for i, data in enumerate(batch):

        b_idx = torch.full((data['indices'].shape[0], 1), i, dtype=torch.int)
        indices_list.append(torch.cat([b_idx, data['indices']], dim=1))
        voxels_list.append(data['voxels'])
        

        current_labels = data['point_labels'] + node_offset
        point_labels_list.append(current_labels)
        

        num_nodes = data['point_labels'].max().item() + 1
        parent = list(range(int(num_nodes)))

        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j: parent[root_i] = root_j

        edge_idx = data['edge_index']
        edge_gt = data['edge_gt']
        
        if edge_idx.shape[1] > 0:

            for j in range(edge_idx.shape[1]):
                if edge_gt[j] == 1:
                    u, v = edge_idx[0, j].item(), edge_idx[1, j].item()
                    union(u, v)
            
            edge_index_list.append(edge_idx + node_offset)
            edge_gt_list.append(edge_gt)
        

        sample_node_gt_ids = torch.tensor([find(k) for k in range(num_nodes)], dtype=torch.long)
        node_gt_ids_list.append(sample_node_gt_ids + nuc_id_offset)

        node_offset += num_nodes
        nuc_id_offset += (num_nodes + 100)

    return {
        "voxels": torch.cat(voxels_list, dim=0),
        "indices": torch.cat(indices_list, dim=0),
        "point_labels": torch.cat(point_labels_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2,0), dtype=torch.long),
        "edge_gt": torch.cat(edge_gt_list, dim=0) if edge_gt_list else torch.zeros(0),
        "node_gt_ids": torch.cat(node_gt_ids_list, dim=0),
        "spatial_shape": max_spatial,
        "batch_size": len(batch)
    }

import json

def test_dataset_stats(h5_path, is_val=False):

    dataset = NucleiDistanceDataset(h5_path, dist_threshold=7)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=nuclei_collate_fn)
    

    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Dataset is empty!")
        return

    total_nodes = batch['point_labels'].max().item() + 1
    node_to_batch = torch.full((total_nodes,), -1, dtype=torch.long)
    node_to_batch[batch['point_labels']] = batch['indices'][:, 0].long()
    

    edge_sample_ids = node_to_batch[batch['edge_index'][0]]
    
    print("\n" + "="*75)
    header = f"{'Sample ID':^10} | {'Total Edges':^12} | {'Pos (Merge)':^12} | {'Pos Ratio':^10} | {'Type':^10}"
    print(header)
    print("-" * 75)
    
    total_pos_global = 0
    total_edges_global = 0

    for b_idx in range(batch['batch_size']):
        mask = (edge_sample_ids == b_idx)
        sample_gt = batch['edge_gt'][mask]
        
        total = sample_gt.numel()
        pos = torch.sum(sample_gt == 1.0).item()
        ratio = (pos / total * 100) if total > 0 else 0
        
        data_type = "VAL" if is_val else "SYNTH"
        print(f"{b_idx:^10} | {total:^12} | {int(pos):^12} | {ratio:^9.2f}% | {data_type:^10}")
        
        total_pos_global += pos
        total_edges_global += total

    print("-" * 75)
    avg_ratio = (total_pos_global / total_edges_global * 100) if total_edges_global > 0 else 0
    print(f"{'TOTAL':^10} | {total_edges_global:^12} | {int(total_pos_global):^12} | {avg_ratio:^9.2f}%")
    print("="*75 + "\n")

    if is_val:
        print("Checking Val Mappings...")
        with h5py.File(h5_path, 'r') as f:

            first_sid = list(f['mappings'].keys())[0]
            mapping_str = f['mappings'][first_sid][()]
            mapping = json.loads(mapping_str)
            print(f"Sample {first_sid} has {len(mapping)} fragments mapped to GT cells.")

            valid_maps = sum(1 for v in mapping.values() if v != -1)
            print(f"Fragments mapped to actual cells: {valid_maps}")

if __name__ == "__main__":

    SYNTH_H5 = "./data/train.h5"
    print("Testing Synthetic Training Set:")
    test_dataset_stats(SYNTH_H5, is_val=False)

    VAL_H5 = "./data/val.h5"
    print("Testing Real Validation Set:")
    test_dataset_stats(VAL_H5, is_val=True)