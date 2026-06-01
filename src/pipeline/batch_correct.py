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
import torch.multiprocessing as mp
from scipy.ndimage import find_objects
import numpy as np

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.utils.graph.apply_offsets import apply_shift_with_padding, apply_shift_with_padding_seg, load_offsets, restore_original_segmentation
from src.models.graph.sparseUNet.Graph_Model import SubNucleiAggregationNet
from src.datasets.graph.sample_points import gpu_fps, get_boundary
from src.utils.graph.graph_operations import watershed_oversegmentation, UnionFind, get_geometry_constrained_graph
from src.models.sdf.unet_sdf import SDFWrinkleNet

global_graph_model = None
global_sdf_model = None
global_args = None
global_offsets = None

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

def worker_init(args, offsets_dict):
    global global_graph_model, global_sdf_model, global_args, global_offsets
    device = torch.device(args.device)
    

    graph_model = SubNucleiAggregationNet(unet_planes=[32, 64, 128], gcn_hidden=args.gcn_hidden).to(device)
    graph_model.load_state_dict(torch.load(args.graph_model_path, map_location=device, weights_only=False))
    graph_model.eval()
    

    sdf_model = SDFWrinkleNet(in_channels=2, base_feat=args.base_feat).to(device)
    sdf_model.load_state_dict(torch.load(args.sdf_model_path, map_location=device, weights_only=True), strict=False)
    sdf_model.eval()
    
    global_graph_model = graph_model
    global_sdf_model = sdf_model
    global_args = args
    global_offsets = offsets_dict

def calculate_iou_sparse(mask_data1, mask_data2):
    
    sl1, m1 = mask_data1['slice'], mask_data1['mask']
    sl2, m2 = mask_data2['slice'], mask_data2['mask']
    

    z_start = max(sl1[0].start, sl2[0].start)
    z_end = min(sl1[0].stop, sl2[0].stop)
    y_start = max(sl1[1].start, sl2[1].start)
    y_end = min(sl1[1].stop, sl2[1].stop)
    x_start = max(sl1[2].start, sl2[2].start)
    x_end = min(sl1[2].stop, sl2[2].stop)
    
    if z_start >= z_end or y_start >= y_end or x_start >= x_end:
        return 0.0, 0.0, 0.0
    

    dz1, dy1, dx1 = z_start - sl1[0].start, y_start - sl1[1].start, x_start - sl1[2].start
    dz2, dy2, dx2 = z_start - sl2[0].start, y_start - sl2[1].start, x_start - sl2[2].start
    
    d, h, w = z_end - z_start, y_end - y_start, x_end - x_start
    
    sub_m1 = m1[dz1:dz1+d, dy1:dy1+h, dx1:dx1+w]
    sub_m2 = m2[dz2:dz2+d, dy2:dy2+h, dx2:dx2+w]
    

    intersection = np.logical_and(sub_m1, sub_m2).sum()
    if intersection == 0:
        return 0.0, 0.0, 0.0
        
    vol1 = mask_data1['volume']
    vol2 = mask_data2['volume']
    union = vol1 + vol2 - intersection
    
    iou = intersection / union if union > 0 else 0
    ratio1 = intersection / vol1 if vol1 > 0 else 0
    ratio2 = intersection / vol2 if vol2 > 0 else 0
    
    return iou, ratio1, ratio2

def process_task(task):
    fid, img_path, seg_path, correct_path, output_dir = task
    graph_model, sdf_model, args, offsets = global_graph_model, global_sdf_model, global_args, global_offsets
    device = torch.device(args.device)
    crop_size = args.crop_size
    half_size = crop_size // 2
    
    try:
        # --- Step 1: Align ---
        img = tiff.imread(img_path); seg = tiff.imread(seg_path); correct = tiff.imread(correct_path)
        original_shape = img.shape
        seg[seg > 0] = 1
        
        off_data = offsets.get(fid) or offsets.get(int(fid) if fid and fid.isdigit() else None)
        
        aligned_img, aligned_seg, aligned_correct, shifts = img, seg, correct, None
        if off_data:
            misaligned_indices = set(off_data.get('bad_slices', []))
            
            for s in off_data.get('relative_shifts', []):
                idx = s.get('to_slice')
                if idx is None: continue
                if s.get('status') == 'skip' or s.get('dx', 0) != 0 or s.get('dy', 0) != 0:
                    misaligned_indices.add(idx)

            for idx in misaligned_indices:
                if 0 <= idx < seg.shape[0]:
                    seg[idx] = 0
            
            flow_volume = []
            for s in off_data['relative_shifts']:
                if s['status'] == 'skip': 
                    flow_volume.append({'skip': True})
                else: 
                    flow_volume.append({'dx': s['dx'], 'dy': s['dy'], 'ref_index': s['from_slice']})
            
            aligned_img, shifts, _ = apply_shift_with_padding(img, flow_volume, list(misaligned_indices))
            aligned_seg = apply_shift_with_padding_seg(seg, shifts)
            aligned_correct = apply_shift_with_padding_seg(correct, shifts)

        # --- Step 2: Watershed Oversegmentation ---
        over_seg = watershed_oversegmentation(aligned_seg, min_size=20, min_distance=7)
        # --- Step 3: Graph Aggregation ---
        edge_registry = {} 
        active_coords = np.argwhere(over_seg > 0)
        if active_coords.size == 0: 
            return {"fid": fid, "status": "no_active_segments", "tp": 0, "fp": 0, "fn": 0}
        
        z_min, y_min, x_min = active_coords.min(axis=0)
        z_max, y_max, x_max = active_coords.max(axis=0)

        pad = args.block_size // 4
        z_start, z_end = max(0, z_min - pad), min(over_seg.shape[0], z_max + pad)
        y_start, y_end = max(0, y_min - pad), min(over_seg.shape[1], y_max + pad)
        x_start, x_end = max(0, x_min - pad), min(over_seg.shape[2], x_max + pad)

        z_range = range(z_start, max(z_start + 1, z_end - args.block_size + 1), args.stride)
        y_range = range(y_start, max(y_start + 1, y_end - args.block_size + 1), args.stride)
        x_range = range(x_start, max(x_start + 1, x_end - args.block_size + 1), args.stride)
        
        for z in z_range:
            for y in y_range:
                for x in x_range:
                    local_mask = over_seg[z:z+args.block_size, 
                                         y:y+args.block_size, 
                                         x:x+args.block_size].copy()
                    
                    if not np.any(local_mask > 0): continue
                    
                    boundary_region = np.ones_like(local_mask, dtype=bool)
                    boundary_region[1:-1, 1:-1, 1:-1] = False
                    boundary_labels = np.unique(local_mask[boundary_region])
                    boundary_labels = boundary_labels[boundary_labels > 0]
                    if len(boundary_labels) > 0:
                        local_mask[np.isin(local_mask, boundary_labels)] = 0
                    
                    if np.sum(local_mask > 0) < 128: continue
                    
                    local_boundary = get_boundary(local_mask)
                    coords_win = np.argwhere(local_boundary)
                    labels_win = local_mask[local_boundary]
                    
                    if len(coords_win) < 128: continue

                    if len(coords_win) > args.target_n:
                        idx_pt = gpu_fps(coords_win, args.target_n)
                        coords_win, labels_win = coords_win[idx_pt], labels_win[idx_pt]

                    edge_index_np, id_map = get_geometry_constrained_graph(
                        coords_win, labels_win, dist_threshold=args.dist_threshold
                    )
                    if edge_index_np.shape[1] == 0: continue

                    voxels_tensor = torch.ones((len(coords_win), 1), dtype=torch.float32).to(device)
                    edge_index_tensor = torch.from_numpy(edge_index_np).long().to(device)
                    point_labels_encoded = np.array([id_map[lid] for lid in labels_win])
                    point_labels_tensor = torch.from_numpy(point_labels_encoded).long().to(device)
                    _, inv_idx = torch.unique(point_labels_tensor, return_inverse=True)
                    
                    indices = torch.zeros((voxels_tensor.shape[0], 4), dtype=torch.int32).to(device)
                    indices[:, 1:] = torch.from_numpy(coords_win).to(device).int()

                    with torch.no_grad():
                        logits, _, _= graph_model(voxels_tensor, indices, 1, local_mask.shape, inv_idx, edge_index_tensor)
                        probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

                    rev_id_map = {v: k for k, v in id_map.items()}
                    for i in range(len(probs)):
                        u_id = rev_id_map[edge_index_np[0, i]]
                        v_id = rev_id_map[edge_index_np[1, i]]
                        edge_key = tuple(sorted([int(u_id), int(v_id)]))
                        p_val = float(probs[i])
                        
                        if edge_key not in edge_registry:
                            edge_registry[edge_key] = [p_val, 1, p_val]
                        else:
                            edge_registry[edge_key][0] += p_val
                            edge_registry[edge_key][1] += 1
                            if p_val < edge_registry[edge_key][2]:
                                edge_registry[edge_key][2] = p_val

        unique_fragments = np.unique(over_seg); unique_fragments = unique_fragments[unique_fragments > 0]
        uf = UnionFind(unique_fragments)
        for (id1, id2), (sum_p, count, min_p) in edge_registry.items():
            if sum_p / count > 0.5: uf.union(id1, id2)
        
        lut = np.arange(over_seg.max() + 1, dtype=np.uint32)
        for lid in unique_fragments: lut[lid] = uf.find(lid)
        aggregated_seg = lut[over_seg]
        
        unique_agg = np.unique(aggregated_seg); unique_agg = unique_agg[unique_agg > 0]
        relabel_lut = np.zeros(aggregated_seg.max() + 1, dtype=np.uint32)
        for new_id, old_id in enumerate(unique_agg, start=1): relabel_lut[old_id] = new_id
        aggregated_seg = relabel_lut[aggregated_seg].astype(np.int32)
        # aggregated_seg = over_seg
        # ---------------------------------------------------------
        # Step 4: SDF-Net Completion + Confidence-based NMS
        # ---------------------------------------------------------
        predictions = []
        object_slices = find_objects(aggregated_seg)
        
        for idx, slices in enumerate(object_slices, start=1):
            if slices is None: continue
            z_slice, y_slice, x_slice = slices
            cz, cy, cx = (z_slice.start + z_slice.stop)//2, (y_slice.start + y_slice.stop)//2, (x_slice.start + x_slice.stop)//2
            

            z_start, z_end = max(0, cz - half_size), min(aligned_img.shape[0], cz + half_size)
            y_start, y_end = max(0, cy - half_size), min(aligned_img.shape[1], cy + half_size)
            x_start, x_end = max(0, cx - half_size), min(aligned_img.shape[2], cx + half_size)
            
            img_crop = aligned_img[z_start:z_end, y_start:y_end, x_start:x_end]

            seg_crop = (aggregated_seg[z_start:z_end, y_start:y_end, x_start:x_end] == idx).astype(np.float32)
            
            curr_d, curr_h, curr_w = img_crop.shape

            if curr_d != crop_size or curr_h != crop_size or curr_w != crop_size:
                pad_d, pad_h, pad_w = (0, crop_size-curr_d), (0, crop_size-curr_h), (0, crop_size-curr_w)
                img_crop = np.pad(img_crop, (pad_d, pad_h, pad_w), mode='constant')
                seg_crop = np.pad(seg_crop, (pad_d, pad_h, pad_w), mode='constant')
                
            img_tensor = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0).to(device)
            img_tensor = img_tensor / 255.0
            seg_tensor = torch.from_numpy(seg_crop).float().unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():

                pred_sdf, pred_mask, conf = sdf_model(img_tensor, seg_tensor)
                

                pred_prob = torch.sigmoid(pred_mask).cpu().numpy()[0, 0]
                pred_bin = (pred_prob > 0.5).astype(float)
                conf_val = conf.item() 
                
            # Crop back to valid region
            valid_mask = pred_bin[:curr_d, :curr_h, :curr_w].astype(bool)
            volume = np.sum(valid_mask)
            
            if volume > 100: 
                predictions.append({
                    'id': idx, 
                    'mask': valid_mask,
                    'slice': (slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end)),
                    'volume': volume,
                    'conf': conf_val 
                })

        predictions.sort(key=lambda x: x['conf'], reverse=True)
        
        keep_list = []
        suppressed_indices = set()
        
        for i in range(len(predictions)):
            if i in suppressed_indices:
                continue
            
            current_cand = predictions[i]
            keep_list.append(current_cand)
            
            for j in range(i + 1, len(predictions)):
                if j in suppressed_indices:
                    continue
                other_cand = predictions[j]
                iou, ratio_curr, ratio_other = calculate_iou_sparse(current_cand, other_cand)

                if iou > 0.1 or ratio_other > 0.01:
                    suppressed_indices.add(j)

        final_seg = np.zeros_like(aggregated_seg)
        new_label_id = 1
        
        for cand in keep_list:
            sl = cand['slice']
            local_mask = cand['mask']
            
            target_region = final_seg[sl[0], sl[1], sl[2]]
            
            mask_to_write = local_mask & (target_region == 0)
            
            if np.any(mask_to_write):
                target_region[mask_to_write] = new_label_id
                final_seg[sl[0], sl[1], sl[2]] = target_region
                new_label_id += 1
        # final_seg = over_seg
        # --- Step 5: Restore ---
        if shifts is not None:
            restored_seg = restore_original_segmentation(final_seg, shifts, original_shape)
        else:
            restored_seg = final_seg
            
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, f'{fid}_final_corrected.tiff')
        # tiff.imwrite(final_path, restored_seg.astype(np.uint8))
        
        # --- Step 6: Calculate Metrics ---
        tp, fp, fn, fp_ids, fn_ids = compute_single_image_metrics_fast(restored_seg, correct, args.iou_threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        # return {"fid": fid, "status": "success", "output_path": final_path, "nuclei_count": len(np.unique(final_seg)), 
        #         "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), 
        #         "recall": round(recall, 4), "f1": round(f1, 4), "accuracy": round(accuracy, 4)}
        return {"fid": fid, "status": "success", "output_path": final_path, "nuclei_count": new_label_id - 1, 
                "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), 
                "recall": round(recall, 4), "f1": round(f1, 4), "accuracy": round(accuracy, 4)}
        
    except Exception as e:
        return {"fid": fid, "status": "error", "msg": traceback.format_exc()}

def main():
    parser = argparse.ArgumentParser()
    # Data Paths
    parser.add_argument('--base_dir', type=str, required=True, help="")
    parser.add_argument('--output_dir', type=str, required=True, help="")
    parser.add_argument('--output_json', type=str, required=True, help="")
    parser.add_argument('--offsets_json', type=str, default="slice_offsets.json", help="")
    parser.add_argument('--seg_filter', type=str, default='all', choices=['ms', 'not_ms', 'all'], help="")
    
    # Model Paths
    parser.add_argument('--graph_model_path', type=str, required=True, help="Path to graph model checkpoint (.pth)")
    parser.add_argument('--sdf_model_path', type=str, required=True, help="Path to SDF model checkpoint (.pth)")

    # Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--gcn_hidden', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=128) 
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--target_n', type=int, default=50000)
    parser.add_argument('--dist_threshold', type=float, default=7.0)
    parser.add_argument('--base_feat', type=int, default=32)
    parser.add_argument('--crop_size', type=int, default=128, help="Size of the crop for SDF completion (default: 128)")
    parser.add_argument('--iou_threshold', type=float, default=0.75, help="IoU threshold for TP/FN calculation")
    parser.add_argument('--num_workers', type=int, default=2, help="")
    
    args = parser.parse_args()

    offsets_path = args.offsets_json if os.path.isabs(args.offsets_json) else os.path.join(args.base_dir, args.offsets_json)
    offsets = load_offsets(offsets_path)
    img_dir = os.path.join(args.base_dir, "img")
    seg_dir = os.path.join(args.base_dir, "match_seg_new")
    correct_dir = os.path.join(args.base_dir, "correct")

    img_map = {re.search(r'\d+', f).group(): f for f in os.listdir(img_dir) if re.search(r'\d+', f) and f.endswith('.tiff')}
    correct_map = {re.search(r'\d+', f).group(): f for f in os.listdir(correct_dir) if re.search(r'\d+', f) and f.endswith('.tiff')}
    tasks = []
    for f_name in os.listdir(seg_dir):
        if not f_name.endswith('.tiff'): continue
        

        if args.seg_filter == 'ms' and 'ms' not in f_name:
            continue
        if args.seg_filter == 'not_ms' and 'ms' in f_name:
            continue
            
        match = re.search(r'\d+', f_name)
        if not match: continue
        fid = match.group()
        if fid in img_map and fid in correct_map:
            img_path = os.path.join(img_dir, img_map[fid])
            seg_path = os.path.join(seg_dir, f_name)
            correct_path = os.path.join(correct_dir, correct_map[fid])

            if 'ms' in f_name:
                error_type = 'ms'
            elif 'split' in args.base_dir.lower() or 'split' in f_name.lower():
                error_type = 'split'
            else:
                error_type = 'merge'
            error_output_dir = os.path.join(args.output_dir, error_type)
            os.makedirs(error_output_dir, exist_ok=True)
            tasks.append((fid, img_path, seg_path, correct_path, error_output_dir))

    print(f"")

    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(processes=args.num_workers, initializer=worker_init, initargs=(args, offsets))
    
    results = []
    for res in tqdm(pool.imap_unordered(process_task, tasks), total=len(tasks), desc="Processing"):
        results.append(res)

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    no_active_count = sum(1 for r in results if r["status"] == "no_active_segments")

    valid = [r for r in results if r["status"] == "success"]
    total_tp = sum(r.get("tp", 0) for r in valid)
    total_fp = sum(r.get("fp", 0) for r in valid)
    total_fn = sum(r.get("fn", 0) for r in valid)
    

    g_pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1 = 2 * g_pre * g_rec / (g_pre + g_rec) if (g_pre + g_rec) > 0 else 0

    g_acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    final_report = {
        "summary": {
            "total_files": len(tasks),
            "success": success_count,
            "error": error_count,
            "no_active_segments": no_active_count,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(g_pre, 4),
            "recall": round(g_rec, 4),
            "f1": round(g_f1, 4),
            "accuracy": round(g_acc, 4)
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
    print("-" * 30)
    print(f"{'Total Files':<15} | {len(tasks)}")
    print(f"{'Success':<15} | {success_count}")
    print(f"{'Error':<15} | {error_count}")
    print(f"{'No Active':<15} | {no_active_count}")
    print("="*40)
    print(f"")
    print(f"")
    print(f"")

if __name__ == "__main__":
    main()
