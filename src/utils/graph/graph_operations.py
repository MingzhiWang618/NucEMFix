import torch
import numpy as np
from tqdm import tqdm
import cc3d
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def watershed_oversegmentation(seg, min_size=100, min_distance=10):
    unique_ids = np.unique(seg)
    unique_ids = unique_ids[unique_ids > 0]
    out = np.zeros_like(seg, dtype=np.int32)
    label_offset = 1

    for id_val in tqdm(unique_ids, desc="Watershed Instances", leave=False):
        id_mask = (seg == id_val)
        cc = cc3d.connected_components(id_mask, connectivity=6)

        for cc_label in range(1, cc.max() + 1):
            region_mask = (cc == cc_label)
            if np.count_nonzero(region_mask) < 10:
                continue

            coords = np.argwhere(region_mask)
            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0) + 1
            seg_crop = region_mask[zmin:zmax, ymin:ymax, xmin:xmax]
            distance_map = distance_transform_edt(seg_crop)

            local_max = peak_local_max(distance_map, min_distance=min_distance,
                                       labels=seg_crop, exclude_border=False)
            if len(local_max) == 0:
                center = np.array(seg_crop.shape) // 2
                local_max = np.array([center])

            markers = np.zeros_like(seg_crop, dtype=np.int32)
            for i, (z, y, x) in enumerate(local_max, start=1):
                markers[z, y, x] = i

            labels_ws = watershed(-distance_map, markers=markers, mask=seg_crop)

            region_sizes = [np.count_nonzero(labels_ws == sub_id)
                            for sub_id in range(1, labels_ws.max() + 1)]
            if not region_sizes:
                continue

            median_size = np.median(region_sizes)
            min_region_size = max(min_size, int(0.3 * median_size))

            final_labels = np.zeros_like(labels_ws, dtype=np.int32)
            curr_label = 1
            for sub_id in range(1, labels_ws.max() + 1):
                mask_sub = (labels_ws == sub_id)
                if np.count_nonzero(mask_sub) >= min_region_size:
                    final_labels[mask_sub] = curr_label
                    curr_label += 1

            unprocessed = (final_labels == 0) & (labels_ws > 0)
            if np.any(unprocessed):
                struct = np.ones((3, 3, 3))
                dilated = binary_dilation(final_labels > 0, structure=struct)
                neighbors = final_labels * dilated
                for z, y, x in np.argwhere(unprocessed):
                    local = neighbors[max(0, z-1):z+2, max(0, y-1):y+2, max(0, x-1):x+2]
                    local = local[local > 0]
                    if local.size > 0:
                        values, counts = np.unique(local, return_counts=True)
                        final_labels[z, y, x] = values[np.argmax(counts)]

            for sub_id in range(1, final_labels.max() + 1):
                mask_sub = (final_labels == sub_id)
                if np.count_nonzero(mask_sub) >= min_size:
                    out[zmin:zmax, ymin:ymax, xmin:xmax][mask_sub] = label_offset
                    label_offset += 1
    return out


class UnionFind:
    def __init__(self, nodes):
        self.parent = {int(node): int(node) for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


@torch.no_grad()
def get_geometry_constrained_graph(coords, labels, dist_threshold=10.0, max_pts_for_dist=500):
    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    id_map = {old_id: i for i, old_id in enumerate(unique_ids)}

    obj_pts_dict = {uid: torch.from_numpy(coords[labels == uid]).float() for uid in unique_ids}
    obj_bboxes = {uid: np.concatenate([coords[labels == uid].min(axis=0),
                                       coords[labels == uid].max(axis=0)]) for uid in unique_ids}

    edge_index = []
    uids_list = list(unique_ids)
    for i in range(len(uids_list)):
        for j in range(i + 1, len(uids_list)):
            id1, id2 = uids_list[i], uids_list[j]
            b1, b2 = obj_bboxes[id1], obj_bboxes[id2]
            bbox_dist = np.sqrt(max(0, b2[0]-b1[3])**2 +
                                max(0, b2[1]-b1[4])**2 +
                                max(0, b2[2]-b1[5])**2)
            if bbox_dist > dist_threshold:
                continue

            p1, p2 = obj_pts_dict[id1], obj_pts_dict[id2]
            if len(p1) > max_pts_for_dist:
                p1 = p1[torch.linspace(0, len(p1)-1, max_pts_for_dist).long()]
            if len(p2) > max_pts_for_dist:
                p2 = p2[torch.linspace(0, len(p2)-1, max_pts_for_dist).long()]

            dists = torch.cdist(p1.unsqueeze(0), p2.unsqueeze(0), p=2).squeeze(0)
            if torch.any(dists <= dist_threshold):
                edge_index.append([id_map[id1], id_map[id2]])

    return np.array(edge_index).T if edge_index else np.empty((2, 0), dtype=int), id_map
