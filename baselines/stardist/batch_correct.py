import os
import re
import sys
import json
import numpy as np
import tifffile as tiff
from tqdm import tqdm
import argparse
import traceback
import torch.multiprocessing as mp
from stardist.models import StarDist3D

from src.utils.graph.apply_offsets import (

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

    apply_shift_with_padding, 
    apply_shift_with_padding_seg, 
    load_offsets
)

def filter_and_evaluate(pred, gt, iou_threshold=0.75, min_size=5):
    """
    """
    if pred.max() == 0:
        return 0, 0, len(np.unique(gt[gt > 0]))

    overlapped_ids = np.unique(pred[gt > 0])
    overlapped_ids = overlapped_ids[overlapped_ids > 0]

    ids, counts = np.unique(pred, return_counts=True)
    id_to_count = dict(zip(ids, counts))

    filtered_pred = np.zeros_like(pred, dtype=np.uint32)
    for pid in overlapped_ids:
        if id_to_count.get(pid, 0) >= min_size:
            filtered_pred[pred == pid] = pid

    pred_ids = np.unique(filtered_pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        return 0, len(pred_ids), 0

    offset = int(gt.max() + 1)
    combined = filtered_pred.astype(np.uint64) * offset + gt.astype(np.uint64)
    mask = (filtered_pred > 0) & (gt > 0)
    overlap_values, overlap_counts = np.unique(combined[mask], return_counts=True)

    intersections = {}
    for val, count in zip(overlap_values, overlap_counts):
        p_id, g_id = int(val // offset), int(val % offset)
        intersections[(p_id, g_id)] = int(count)

    f_pred_areas = {int(pid): int(count) for pid, count in zip(*np.unique(filtered_pred[filtered_pred > 0], return_counts=True))}
    gt_areas = {int(gid): int(count) for gid, count in zip(*np.unique(gt[gt > 0], return_counts=True))}

    tp = 0
    matched_gt, matched_pred = set(), set()
    for (p_id, g_id), intersect_vol in intersections.items():
        union_vol = f_pred_areas[p_id] + gt_areas[g_id] - intersect_vol
        if intersect_vol / union_vol >= iou_threshold:
            tp += 1
            matched_gt.add(g_id)
            matched_pred.add(p_id)

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    
    return tp, fp, fn

global_model = None
global_args = None
global_offsets = None

def worker_init(args, offsets_dict):
    global global_model, global_args, global_offsets
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.replace("cuda:", "")
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    global_model = StarDist3D(None, name=args.model_name, basedir=args.model_dir)
    global_args = args
    global_offsets = offsets_dict

def process_and_eval_task(task):
    fid, img_path, correct_path = task
    model, args, offsets = global_model, global_args, global_offsets
    
    try:
        img = tiff.imread(img_path)
        correct = tiff.imread(correct_path)
        

        off_data = offsets.get(fid) or offsets.get(int(fid) if fid.isdigit() else None)
        aligned_img, aligned_correct = img, correct
        if off_data:
            bad_slices = off_data.get('bad_slices', [])
            flow_volume = [{'skip': True} if s['status'] == 'skip' else 
                           {'dx': s['dx'], 'dy': s['dy'], 'ref_index': s['from_slice']} 
                           for s in off_data['relative_shifts']]
            aligned_img, shifts, _ = apply_shift_with_padding(img, flow_volume, bad_slices)
            aligned_correct = apply_shift_with_padding_seg(correct, shifts)

        coords = np.argwhere(aligned_correct > 0)
        if coords.size == 0: return {"fid": fid, "status": "empty_gt", "tp": 0, "fp": 0, "fn": 0}
        
        crop_slice = (slice(max(0, coords[:,0].min()-args.padding), coords[:,0].max()+args.padding),
                      slice(max(0, coords[:,1].min()-args.padding), coords[:,1].max()+args.padding),
                      slice(max(0, coords[:,2].min()-args.padding), coords[:,2].max()+args.padding))

        img_crop = aligned_img[crop_slice]
        mi, ma = np.percentile(img_crop, 1), np.percentile(img_crop, 99.8)
        img_norm = (img_crop - mi) / (ma - mi + 1e-20)
        pred_labels, _ = model.predict_instances(img_norm)

        tp, fp, fn = filter_and_evaluate(pred_labels, aligned_correct[crop_slice], args.iou_threshold, min_size=5)
        
        return {"fid": fid, "status": "success", "tp": tp, "fp": fp, "fn": fn}
    except Exception:
        return {"fid": fid, "status": "error", "msg": traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--offsets_json', type=str, default="slice_offsets.json")

    parser.add_argument('--seg_filter', type=str, default='all', choices=['ms', 'not_ms', 'all'])
    parser.add_argument('--iou_threshold', type=float, default=0.75)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default="cuda:6")
    parser.add_argument('--padding', type=int, default=32)
    parser.add_argument('--model_dir', type=str, default="./baselines/stardist/models")
    parser.add_argument('--model_name', type=str, default="stardist_fafb")
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
    results = [res for res in tqdm(pool.imap_unordered(process_and_eval_task, tasks), total=len(tasks))]
    pool.close()

    valid = [r for r in results if r["status"] == "success"]
    total_tp = sum(r["tp"] for r in valid)
    total_fp = sum(r["fp"] for r in valid)
    total_fn = sum(r["fn"] for r in valid)
    
    pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    report = {"summary": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": pre, "recall": rec, "f1": f1, "accuracy": acc}, "per_file": results}
    with open(args.output_json, 'w') as f: json.dump(report, f, indent=4)

    print("\n" + "="*45 + f"\n[Filter: {args.seg_filter}]\nPrecision: {pre:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\nAccuracy: {acc:.4f}\n" + "="*45)

if __name__ == "__main__":
    main()