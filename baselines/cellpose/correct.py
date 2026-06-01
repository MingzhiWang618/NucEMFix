import os
import sys
import re
import numpy as np
import tifffile as tiff
from cellpose import models, io

from src.utils.graph.apply_offsets import (

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

    apply_shift_with_padding, 
    apply_shift_with_padding_seg, 
    load_offsets, 
    restore_original_segmentation
)

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    pass  # set paths via argparse (see __main__ block)

    print(f"Loading Cellpose model from: {model_path}")

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    print("Loading and aligning data...")
    img = tiff.imread(img_path)
    gt = tiff.imread(seg_path)
    original_shape = img.shape
    
    filename = os.path.basename(img_path)
    fid = re.search(r'\d+', filename).group()
    offsets = load_offsets(offsets_json)
    off_data = offsets.get(fid) or offsets.get(int(fid))
    
    if off_data:
        bad_slices = off_data.get('bad_slices', [])
        flow_volume = [{'skip': True} if s['status'] == 'skip' else 
                        {'dx': s['dx'], 'dy': s['dy'], 'ref_index': s['from_slice']} 
                        for s in off_data['relative_shifts']]
        
        aligned_img, shifts, _ = apply_shift_with_padding(img.astype(np.float32), flow_volume, bad_slices)
        aligned_gt = apply_shift_with_padding_seg(gt, shifts)
    else:
        aligned_img, aligned_gt, shifts = img.astype(np.float32), gt, None

    slices = get_bbox_with_padding(aligned_gt, padding=32, shape_limit=aligned_img.shape)
    if slices is None: return
    img_crop = aligned_img[slices]
    gt_crop = aligned_gt[slices]

    print(f"Cellpose 3D Predicting (shape: {img_crop.shape})...")
    

    mask_pred_crop, flows, styles = model.eval(
        img_crop, 
        channels=[0, 0], 
        do_3D=True, 
        anisotropy=1.25,
        diameter=10.0, 
        cellprob_threshold=0.5, 
        flow_threshold=0.4,
        batch_size=16
    )

    print("Finalizing results...")
    

    mask_pred_crop[gt_crop == 0] = 0

    full_aligned_pred = np.zeros(aligned_gt.shape, dtype=np.uint32)
    full_aligned_pred[slices] = mask_pred_crop
    shifts = None 

    if shifts is not None:
        final_restored_seg = restore_original_segmentation(full_aligned_pred, shifts, original_shape)
    else:
        final_restored_seg = full_aligned_pred

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{fid}_cellpose_final.tiff")
    tiff.imwrite(save_path, final_restored_seg.astype(np.uint16), compression='zlib')

    print(f"✅ Cellpose Inference Success! Saved to: {save_path}")

if __name__ == '__main__':
    main()