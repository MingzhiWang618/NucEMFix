import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm

def compute_single_image_stats(pred, gt, iou_threshold=0.5):
    """
    """
    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        return 0, len(pred_ids), 0, 0, len(pred_ids)

    offset = int(gt.max() + 1)
    combined = pred.astype(np.uint64) * offset + gt.astype(np.uint64)
    mask = (pred > 0) & (gt > 0)
    overlap_values, overlap_counts = np.unique(combined[mask], return_counts=True)

    intersections = {}
    for val, count in zip(overlap_values, overlap_counts):
        p_id = int(val // offset)
        g_id = int(val % offset)
        intersections[(p_id, g_id)] = count

    pred_areas = {pid: count for pid, count in zip(*np.unique(pred[pred > 0], return_counts=True))}
    gt_areas = {gid: count for gid, count in zip(*np.unique(gt[gt > 0], return_counts=True))}

    tp = 0
    matched_gt = set()
    matched_pred = set()

    for (p_id, g_id), intersect_vol in intersections.items():
        union_vol = pred_areas[p_id] + gt_areas[g_id] - intersect_vol
        iou = intersect_vol / union_vol

        if iou >= iou_threshold:
            tp += 1
            matched_gt.add(g_id)
            matched_pred.add(p_id)

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)
    
    return tp, fp, fn, len(gt_ids), len(pred_ids)

def evaluate_dataset(pred_dir, gt_dir, iou_threshold=0.5):
    """
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt_instances = 0
    total_pred_instances = 0

    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.tif')])
    
    print(f"📊 Evaluating {len(pred_files)} files...")
    
    for filename in tqdm(pred_files):
        pred_path = os.path.join(pred_dir, filename)
        gt_filename = filename.replace(".tif", "_proofreaded.tif") 
        gt_path = os.path.join(gt_dir, gt_filename)

        if not os.path.exists(gt_path):
            print(f"⚠️ Warning: GT file not found for {filename}, skipping...")
            continue

        pred = tiff.imread(pred_path)
        gt = tiff.imread(gt_path)
        print(f"Evaluating {filename}...")
        tp, fp, fn, n_gt, n_pred = compute_single_image_stats(pred, gt, iou_threshold)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt_instances += n_gt
        total_pred_instances += n_pred

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print(f"FINAL DATASET METRICS (IoU > {iou_threshold})")
    print("-" * 40)
    print(f"Total GT Instances:   {total_gt_instances}")
    print(f"Total Pred Instances: {total_pred_instances}")
    print(f"Total TP:             {total_tp}")
    print(f"Total FP:             {total_fp}")
    print(f"Total FN:             {total_fn}")
    print("-" * 40)
    print(f"Precision:            {precision:.4f}")
    print(f"Recall:               {recall:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print("="*40)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

if __name__ == "__main__":

    PRED_FOLDER = "./results"
    GT_FOLDER = "./data/correct"
    
    evaluate_dataset(PRED_FOLDER, GT_FOLDER, iou_threshold=0.5)