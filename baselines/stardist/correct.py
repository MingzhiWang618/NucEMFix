import os
import sys
import re
import numpy as np
import tifffile as tiff
from stardist.models import StarDist3D
import json

from src.utils.graph.apply_offsets import (

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

    apply_shift_with_padding, 
    apply_shift_with_padding_seg, 
    load_offsets, 
    restore_original_segmentation
)

def filter_labels_by_gt_overlap(pred_labels, gt_crop, min_voxels=20):
    """
    """

    print(f"Input shapes: pred_labels={pred_labels.shape}, gt_crop={gt_crop.shape}")
    print(f"Pred labels stats: min={pred_labels.min()}, max={pred_labels.max()}, unique={np.unique(pred_labels)}")
    print(f"GT crop stats: min={gt_crop.min()}, max={gt_crop.max()}, unique={np.unique(gt_crop)}")
    
    if pred_labels.max() == 0:
        print("No predictions found!")
        return np.zeros_like(pred_labels)
    
    if gt_crop.max() == 0:
        print("No GT found!")
        return np.zeros_like(pred_labels)

    gt_mask = gt_crop > 0
    print(f"GT mask shape: {gt_mask.shape}, sum: {np.sum(gt_mask)}")
    
    overlapped_pred = pred_labels[gt_mask]
    print(f"Overlapped pred stats: min={overlapped_pred.min()}, max={overlapped_pred.max()}, unique={np.unique(overlapped_pred)}")
    
    overlapped_ids = np.unique(overlapped_pred)
    overlapped_ids = overlapped_ids[overlapped_ids > 0]
    print(f"Overlapped IDs: {overlapped_ids}")

    ids, counts = np.unique(pred_labels, return_counts=True)
    id_to_count = dict(zip(ids, counts))
    print(f"All prediction volumes: {id_to_count}")

    final_mask = np.zeros_like(pred_labels, dtype=np.uint32)
    
    keep_count = 0
    for pid in overlapped_ids:
        volume = id_to_count.get(pid, 0)
        print(f"Checking ID {pid}: volume={volume}, min_voxels={min_voxels}")
        if volume >= min_voxels:
            final_mask[pred_labels == pid] = pid
            keep_count += 1
            
    print(f"GT Overlapped: {len(overlapped_ids)}, After size filter: {keep_count}")
    return final_mask

def manual_normalize(x, pmin=1, pmax=99.8, eps=1e-20):
    mi = np.percentile(x, pmin)
    ma = np.percentile(x, pmax)
    return (x - mi) / (ma - mi + eps)

def get_bbox_with_padding(mask, padding=32, shape_limit=None):
    coords = np.argwhere(mask > 0)
    if coords.size == 0: return None
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    
    z_start, z_end = max(0, z_min - padding), min(shape_limit[0], z_max + padding)
    y_start, y_end = max(0, y_min - padding), min(shape_limit[1], y_max + padding)
    x_start, x_end = max(0, x_min - padding), min(shape_limit[2], x_max + padding)
    return (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    img_path = "./data/img/sample.tiff"
    seg_path = "./data/seg/sample.tiff"
    offsets_json = "./data/slice_offsets.json"
    output_dir = "./results/stardist"

    model_basedir = './checkpoints/stardist/models'
    model_name = 'stardist_mouse'

    img = tiff.imread(img_path)
    gt = tiff.imread(seg_path) 
    original_shape = img.shape

    fid = re.search(r'\d+', os.path.basename(img_path)).group()
    offsets = load_offsets(offsets_json)
    off_data = offsets.get(fid) or offsets.get(int(fid))
    
    if off_data:
        bad_slices = off_data.get('bad_slices', [])
        flow_volume = [{'skip': True} if s['status'] == 'skip' else 
                       {'dx': s['dx'], 'dy': s['dy'], 'ref_index': s['from_slice']} 
                       for s in off_data['relative_shifts']]
        aligned_img, shifts, _ = apply_shift_with_padding(img, flow_volume, bad_slices)
        aligned_gt = apply_shift_with_padding_seg(gt, shifts)
    else:
        aligned_img, aligned_gt, shifts = img, gt, None

    slices = get_bbox_with_padding(aligned_gt, padding=48, shape_limit=aligned_img.shape)
    if slices is None: return

    img_crop = aligned_img[slices]
    gt_crop = aligned_gt[slices]

    print("StarDist Predicting...")
    model = StarDist3D(None, name=model_name, basedir=model_basedir)
    img_norm = manual_normalize(img_crop)
    pred_labels, _ = model.predict_instances(img_norm, n_tiles=None)

    print("Filtering results by GT overlap and size...")
    final_pred_crop = filter_labels_by_gt_overlap(pred_labels, gt_crop, min_voxels=5)
    

    full_aligned_pred = np.zeros_like(aligned_gt, dtype=np.uint32)
    full_aligned_pred[slices] = final_pred_crop
    if shifts is not None:
        final_restored_seg = restore_original_segmentation(full_aligned_pred, shifts, original_shape)
    else:
        final_restored_seg = full_aligned_pred

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{fid}_stardist_gt_filtered.tiff")
    tiff.imwrite(save_path, full_aligned_pred.astype(np.uint16), compression='zlib')

    print(f"✅ Success! Saved to: {save_path}")
    print(f"Remaining Labels: {len(np.unique(final_restored_seg))-1}")

if __name__ == '__main__':
    main()