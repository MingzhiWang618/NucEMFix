import os
import re
import sys
import json
import torch
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import argparse
import traceback
import torch.nn.functional as F
import torch.multiprocessing as mp
from scipy.ndimage import label as nd_label
from skimage.segmentation import watershed

from baselines.unet.model.model import UNet3D_BC 
from src.utils.graph.apply_offsets import (

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

    apply_shift_with_padding, 
    apply_shift_with_padding_seg, 
    load_offsets, 
    restore_original_segmentation
)

def compute_single_image_metrics_fast(pred, gt, iou_threshold=0.75):
    
    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)
    
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        return 0, len(pred_ids), 0, [int(i) for i in pred_ids], []

    offset = int(gt.max() + 1)
    combined = pred.astype(np.uint64) * offset + gt.astype(np.uint64)
    mask = (pred > 0) & (gt > 0)
    overlap_values, overlap_counts = np.unique(combined[mask], return_counts=True)

    intersections = {}
    for val, count in zip(overlap_values, overlap_counts):
        p_id, g_id = int(val // offset), int(val % offset)
        intersections[(p_id, g_id)] = int(count)

    pred_areas = {int(pid): int(count) for pid, count in zip(*np.unique(pred[pred > 0], return_counts=True))}
    gt_areas = {int(gid): int(count) for gid, count in zip(*np.unique(gt[gt > 0], return_counts=True))}

    tp = 0
    matched_gt, matched_pred = set(), set()
    for (p_id, g_id), intersect_vol in intersections.items():
        union_vol = pred_areas[p_id] + gt_areas[g_id] - intersect_vol
        if intersect_vol / union_vol >= iou_threshold:
            tp += 1
            matched_gt.add(g_id)
            matched_pred.add(p_id)

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    return tp, fp, fn, [int(x) for x in (set(pred_ids) - matched_pred)], [int(x) for x in (set(gt_ids) - matched_gt)]

def post_process_bc(prob_map, core_thresh=0.5, boundary_thresh=0.5, min_size=20):
    """ 
    """
    core_prob = prob_map[1]     # Channel 1: Core
    boundary_prob = prob_map[2] # Channel 2: Boundary
    

    seeds_mask = core_prob > core_thresh
    markers, n = nd_label(seeds_mask)
    

    if n > 0:
        counts = np.bincount(markers.ravel())
        mask_sizes = counts[markers] >= min_size
        markers[~mask_sizes] = 0
        markers, _ = nd_label(markers > 0)
    

    basin = -core_prob + boundary_prob 
    

    eff_mask = (core_prob + boundary_prob) > boundary_thresh
    

    labels = watershed(basin, markers, mask=eff_mask)
    

    if labels.max() > 0:
        final_counts = np.bincount(labels.ravel())

        remove_mask = final_counts < min_size
        remove_mask[0] = False
        

        labels_to_remove = np.where(remove_mask)[0]
        if len(labels_to_remove) > 0:

            mask_to_zero = np.isin(labels, labels_to_remove)
            labels[mask_to_zero] = 0
            
    return labels.astype(np.uint32)
def predict_crop(model, img_norm, device):
    
    z, y, x = img_norm.shape

    pad_z = (16 - z % 16) % 16
    pad_y = (16 - y % 16) % 16
    pad_x = (16 - x % 16) % 16
    
    if pad_z > 0 or pad_y > 0 or pad_x > 0:
        img_norm = np.pad(img_norm, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')

    input_t = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
    return probs[:, :z, :y, :x]

global_model = None
global_args = None
global_offsets = None

def worker_init(args, offsets_dict):
    global global_model, global_args, global_offsets
    device = torch.device(args.device)
    model = UNet3D_BC(in_channels=1, n_classes=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    global_model = model
    global_args = args
    global_offsets = offsets_dict

def process_and_eval_task(task):
    fid, img_path, correct_path = task
    model, args, offsets = global_model, global_args, global_offsets
    device = torch.device(args.device)
    
    try:
        img = tiff.imread(img_path).astype(np.float32)
        correct = tiff.imread(correct_path)
        original_shape = img.shape

        off_data = offsets.get(fid) or offsets.get(int(fid) if fid.isdigit() else None)
        aligned_img, aligned_correct, shifts = img, correct, None
        if off_data:
            bad_slices = off_data.get('bad_slices', [])
            flow_volume = [{'skip': True} if s['status'] == 'skip' else 
                           {'dx': s['dx'], 'dy': s['dy'], 'ref_index': s['from_slice']} 
                           for s in off_data['relative_shifts']]
            aligned_img, shifts, _ = apply_shift_with_padding(img, flow_volume, bad_slices)
            aligned_correct = apply_shift_with_padding_seg(correct, shifts)

        coords = np.argwhere(aligned_correct > 0)
        if coords.size == 0:
            return {"fid": fid, "status": "empty_correct", "tp": 0, "fp": 0, "fn": 0}
            
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        z_s, z_e = max(0, z_min - args.padding), min(aligned_img.shape[0], z_max + args.padding)
        y_s, y_e = max(0, y_min - args.padding), min(aligned_img.shape[1], y_max + args.padding)
        x_s, x_e = max(0, x_min - args.padding), min(aligned_img.shape[2], x_max + args.padding)
        crop_slice = (slice(z_s, z_e), slice(y_s, y_e), slice(x_s, x_e))

        img_crop = aligned_img[crop_slice]
        mi, ma = np.percentile(img_crop, 1), np.percentile(img_crop, 99.8)
        img_norm = (img_crop - mi) / (ma - mi + 1e-20)
        
        prob_map_crop = predict_crop(model, img_norm, device)
        

        pred_labels_crop = post_process_bc(prob_map_crop, core_thresh=args.core_thresh)

        pred_labels_crop[aligned_correct[crop_slice] == 0] = 0 
        

        tp, fp, fn, fp_ids, fn_ids = compute_single_image_metrics_fast(pred_labels_crop, aligned_correct[crop_slice], args.iou_threshold)
        
        return {
            "fid": fid, "status": "success", "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp/(tp+fp), 4) if (tp+fp)>0 else 0,
            "recall": round(tp/(tp+fn), 4) if (tp+fn)>0 else 0,
            "f1": round(2*tp/(2*tp+fp+fn), 4) if (2*tp+fp+fn)>0 else 0
        }
        
    except Exception:
        return {"fid": fid, "status": "error", "msg": traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--offsets_json', type=str, default="slice_offsets.json")
    parser.add_argument('--seg_filter', type=str, default='all', choices=['ms', 'not_ms', 'all'])
    parser.add_argument('--iou_threshold', type=float, default=0.75)
    parser.add_argument('--core_thresh', type=float, default=0.7)
    parser.add_argument('--num_workers', type=int, default=2) 
    parser.add_argument('--device', type=str, default="cuda:6")
    parser.add_argument('--padding', type=int, default=32)
    args = parser.parse_args()

    offsets = load_offsets(os.path.join(args.base_dir, args.offsets_json))
    img_dir = os.path.join(args.base_dir, "img")
    correct_dir = os.path.join(args.base_dir, "correct")

    img_map = {re.search(r'\d+', f).group(): f for f in os.listdir(img_dir) if re.search(r'\d+', f) and f.endswith('.tiff')}
    tasks = []
    for f_name in os.listdir(correct_dir):
        if not f_name.endswith('.tiff'): continue
        if args.seg_filter == 'ms' and 'ms' not in f_name: continue
        if args.seg_filter == 'not_ms' and 'ms' in f_name: continue
        match = re.search(r'\d+', f_name)
        if not match: continue
        fid = match.group()
        if fid in img_map:
            tasks.append((fid, os.path.join(img_dir, img_map[fid]), os.path.join(correct_dir, f_name)))

    print(f"")

    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=args.num_workers, initializer=worker_init, initargs=(args, offsets))
    
    results = []
    for res in tqdm(pool.imap_unordered(process_and_eval_task, tasks), total=len(tasks), desc="Evaluation"):
        results.append(res)

    valid = [r for r in results if r["status"] == "success"]
    total_tp = sum(r["tp"] for r in valid)
    total_fp = sum(r["fp"] for r in valid)
    total_fn = sum(r["fn"] for r in valid)
    

    g_pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1 = 2 * g_pre * g_rec / (g_pre + g_rec) if (g_pre + g_rec) > 0 else 0

    g_acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    final_report = {
        "summary": {
            "total_tp": total_tp, "total_fp": total_fp, "total_fn": total_fn,
            "precision": round(g_pre, 4), "recall": round(g_rec, 4),
            "f1": round(g_f1, 4), "accuracy": round(g_acc, 4)
        },
        "per_file": results
    }

    with open(args.output_json, 'w') as f:
        json.dump(final_report, f, indent=4)
    

    print("\n" + "="*40)
    print(f"{'Global Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Precision':<15} | {g_pre:.4f}")
    print(f"{'Recall':<15} | {g_rec:.4f}")
    print(f"{'F1-Score':<15} | {g_f1:.4f}")
    print(f"{'Accuracy':<15} | {g_acc:.4f}")
    print("="*40)
    print(f"")

if __name__ == "__main__":
    main()