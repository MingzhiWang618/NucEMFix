#!/usr/bin/env python3
# ===================================================

# ===================================================
import os
import re
import json
import numpy as np
import argparse
from tqdm import tqdm
from scipy.ndimage import label as nd_label
import tifffile as tiff

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

def evaluate_segmentation(pred_dir, correct_dir, seg_filter='all', iou_threshold=0.75):
    

    pred_files = {f: os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith('.tiff')}
    correct_files = {f: os.path.join(correct_dir, f) for f in os.listdir(correct_dir) if f.endswith('.tiff')}
    

    common_files = set(pred_files.keys()) & set(correct_files.keys())
    pass
    

    filtered_files = []
    for f_name in common_files:
        if seg_filter == 'all':
            filtered_files.append(f_name)
        elif seg_filter == 'ms' and 'ms' in f_name:
            filtered_files.append(f_name)
        elif seg_filter == 'not_ms' and 'ms' not in f_name:
            filtered_files.append(f_name)
    
    pass
    

    results = []
    for f_name in tqdm(filtered_files, desc=""):
        pred_path = pred_files[f_name]
        correct_path = correct_files[f_name]
        
        try:

            pred = tiff.imread(pred_path)
            correct = tiff.imread(correct_path)
            

            tp, fp, fn, fp_ids, fn_ids = compute_single_image_metrics_fast(pred, correct, iou_threshold)
            

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            
            results.append({
                "filename": f_name,
                "status": "success",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            })
            
        except Exception as e:
            results.append({
                "filename": f_name,
                "status": "error",
                "msg": str(e)
            })
    

    valid_results = [r for r in results if r["status"] == "success"]
    total_tp = sum(r["tp"] for r in valid_results)
    total_fp = sum(r["fp"] for r in valid_results)
    total_fn = sum(r["fn"] for r in valid_results)
    

    g_pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1 = 2 * g_pre * g_rec / (g_pre + g_rec) if (g_pre + g_rec) > 0 else 0

    g_acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    
    final_report = {
        "summary": {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(g_pre, 4),
            "recall": round(g_rec, 4),
            "f1": round(g_f1, 4),
            "accuracy": round(g_acc, 4),
            "total_files": len(filtered_files),
            "success_files": len(valid_results),
            "error_files": len(results) - len(valid_results)
        },
        "per_file": results
    }
    
    return final_report

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--pred_dir', type=str, required=True, help="")
    parser.add_argument('--correct_dir', type=str, required=True, help="")
    parser.add_argument('--output_json', type=str, required=True, help="")
    parser.add_argument('--seg_filter', type=str, default='all', choices=['ms', 'not_ms', 'all'], help="")
    parser.add_argument('--iou_threshold', type=float, default=0.75, help="")
    args = parser.parse_args()
    
    pass
    pass
    pass
    pass
    pass
    

    report = evaluate_segmentation(
        args.pred_dir,
        args.correct_dir,
        args.seg_filter,
        args.iou_threshold
    )
    

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(report, f, indent=4)
    

    print("\n" + "="*40)
    print(f"{'Global Metric':<15} | {'Value':<10}")
    print("-" * 30)
    print(f"{'Precision':<15} | {report['summary']['precision']:.4f}")
    print(f"{'Recall':<15} | {report['summary']['recall']:.4f}")
    print(f"{'F1-Score':<15} | {report['summary']['f1']:.4f}")
    print(f"{'Accuracy':<15} | {report['summary']['accuracy']:.4f}")
    print("-" * 30)
    print(f"{'Total Files':<15} | {report['summary']['total_files']}")
    print(f"{'Success':<15} | {report['summary']['success_files']}")
    print(f"{'Error':<15} | {report['summary']['error_files']}")
    print("="*40)
    pass

if __name__ == "__main__":
    main()
