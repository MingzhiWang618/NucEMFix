import numpy as np
import tifffile as tiff
import json
from tqdm import tqdm

def compute_instance_metrics_original(pred_path, gt_path, output_json, iou_threshold=0.5):
    """
    """
    print(f"Loading files...")
    pred = tiff.imread(pred_path).astype(np.int32)
    gt = tiff.imread(gt_path).astype(np.int32)

    pred_ids = np.unique(pred)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt)
    gt_ids = gt_ids[gt_ids > 0]

    if len(gt_ids) == 0:
        return {"Precision": 0, "Recall": 0, "F1": 0}

    offset = int(gt.max() + 1)
    combined = pred.astype(np.uint64) * offset + gt.astype(np.uint64)
    
    mask = (pred > 0) & (gt > 0)
    overlap_values, overlap_counts = np.unique(combined[mask], return_counts=True)

    intersections = {}
    for val, count in zip(overlap_values, overlap_counts):
        p_id = int(val // offset)
        g_id = int(val % offset)
        intersections[(p_id, g_id)] = int(count)

    pred_areas = {int(pid): int(count) for pid, count in zip(*np.unique(pred[pred > 0], return_counts=True))}
    gt_areas = {int(gid): int(count) for gid, count in zip(*np.unique(gt[gt > 0], return_counts=True))}

    tp = 0
    matched_gt = set()
    matched_pred = set()
    match_details = []

    for (p_id, g_id), intersect_vol in intersections.items():
        union_vol = pred_areas[p_id] + gt_areas[g_id] - intersect_vol
        iou = intersect_vol / union_vol

        if iou >= iou_threshold:
            tp += 1
            matched_gt.add(g_id)
            matched_pred.add(p_id)

            match_details.append({
                "pred_id": p_id,
                "gt_id": g_id,
                "iou": round(float(iou), 4),
                "vol": int(intersect_vol)
            })

    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "summary": {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "GT_Count": len(gt_ids),
            "Pred_Count": len(pred_ids)
        },
        "matched_pairs": match_details,
        "fp_ids": [int(x) for x in (set(pred_ids) - matched_pred)],
        "fn_ids": [int(x) for x in (set(gt_ids) - matched_gt)]
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Metrics saved to {output_json}")
    print(f"F1 Score (Original Logic): {f1:.4f}")

if __name__ == "__main__":
    compute_instance_metrics_original(
        pred_path="./results/pred.tiff", 
        gt_path="./data/correct/gt.tiff", 
        output_json="original_match_results.json",
        iou_threshold=0.75
    )