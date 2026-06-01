import torch
import h5py
import numpy as np
from tqdm import tqdm
from pointnet2_ops import pointnet2_utils
from concurrent.futures import ProcessPoolExecutor
import os

def get_boundary(mask):
    
    boundary = np.zeros_like(mask, dtype=bool)
    boundary[1:-1, 1:-1, 1:-1] = (
        (mask[1:-1, 1:-1, 1:-1] != mask[:-2, 1:-1, 1:-1]) |
        (mask[1:-1, 1:-1, 1:-1] != mask[2:, 1:-1, 1:-1]) |
        (mask[1:-1, 1:-1, 1:-1] != mask[1:-1, :-2, 1:-1]) |
        (mask[1:-1, 1:-1, 1:-1] != mask[1:-1, 2:, 1:-1]) |
        (mask[1:-1, 1:-1, 1:-1] != mask[1:-1, 1:-1, :-2]) |
        (mask[1:-1, 1:-1, 1:-1] != mask[1:-1, 1:-1, 2:])
    )
    boundary = boundary & (mask > 0)
    return boundary

def gpu_fps(coords, n_samples):
    
    if len(coords) <= n_samples:
        return np.arange(len(coords))
    
    points_tensor = torch.from_numpy(coords).float().cuda().unsqueeze(0).contiguous()
    

    func = getattr(pointnet2_utils, 'furthest_point_sample', 
                   getattr(pointnet2_utils, 'farthest_point_sample', None))
    
    if func is None:
        raise ImportError("Could not find FPS function in pointnet2_utils")
        
    fps_idx = func(points_tensor, n_samples)
    return fps_idx.squeeze(0).cpu().numpy().astype(np.int64)

def process_single_sample(sid, mask_data, target_n):
    """ 
    """
    boundary_mask = get_boundary(mask_data)
    all_coords = np.argwhere(boundary_mask)
    all_labels = mask_data[boundary_mask]
    
    if len(all_coords) == 0:
        return sid, None, None

    if len(all_coords) <= target_n:
        return sid, all_coords.astype(np.int16), all_labels.astype(np.int32)
    
    return sid, all_coords, all_labels

def fast_process_h5(h5_path, target_n=50000, min_per_obj=128, num_workers=8):

    with h5py.File(h5_path, 'a') as f:
        if 'sampled_coords' in f:
            pass
            del f['sampled_coords']
        if 'sampled_labels' in f:
            pass
            del f['sampled_labels']
        
        f.create_group('sampled_coords')
        f.create_group('sampled_labels')
        
        all_sids = list(f['masks'].keys())

    pass

    batch_size = 40
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(all_sids), batch_size), desc="Overall Progress"):
            chunk_sids = all_sids[i:i + batch_size]
            

            chunk_masks = []
            with h5py.File(h5_path, 'r') as f_read:
                for sid in chunk_sids:
                    chunk_masks.append(f_read['masks'][sid][:])
            

            futures = [executor.submit(process_single_sample, sid, mask, target_n) 
                       for sid, mask in zip(chunk_sids, chunk_masks)]
            

            with h5py.File(h5_path, 'a') as f_write:
                for future in futures:
                    sid, coords, labels = future.result()
                    if coords is None: continue

                    if coords.dtype == np.int16:

                        f_write['sampled_coords'].create_dataset(sid, data=coords)
                        f_write['sampled_labels'].create_dataset(sid, data=labels)
                    else:

                        unique_ids = np.unique(labels)
                        sampled_indices = []
                        pool_mask = np.ones(len(coords), dtype=bool)

                        for uid in unique_ids:
                            obj_idx = np.where(labels == uid)[0]
                            if len(obj_idx) <= min_per_obj:
                                sampled_indices.append(obj_idx)
                                pool_mask[obj_idx] = False
                            else:
                                local_fps = gpu_fps(coords[obj_idx], min_per_obj)
                                g_idx = obj_idx[local_fps]
                                sampled_indices.append(g_idx)
                                pool_mask[g_idx] = False
                        
                        current_n = sum(len(x) for x in sampled_indices)
                        budget_left = target_n - current_n
                        if budget_left > 0:
                            p_idx = np.where(pool_mask)[0]
                            if len(p_idx) > budget_left:
                                extra = gpu_fps(coords[p_idx], budget_left)
                                sampled_indices.append(p_idx[extra])
                            else:
                                sampled_indices.append(p_idx)
                        
                        f_idx = np.concatenate(sampled_indices)
                        f_write['sampled_coords'].create_dataset(sid, data=coords[f_idx].astype(np.int16))
                        f_write['sampled_labels'].create_dataset(sid, data=labels[f_idx].astype(np.int32))
            

            torch.cuda.empty_cache()

if __name__ == "__main__":
    H5_FILE = "./data/synthetic_dataset.h5"

    fast_process_h5(H5_FILE, num_workers=8)