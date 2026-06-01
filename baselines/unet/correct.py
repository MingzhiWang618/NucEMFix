import os
import sys
import re
import torch
import numpy as np
import tifffile as tiff
import torch.nn.functional as F
from tqdm import tqdm
from skimage.segmentation import watershed
from scipy.ndimage import label as nd_label

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

def manual_normalize(x, pmin=1, pmax=99.8, axis=None, eps=1e-20):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
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

def predict_large_image(model, img, device, patch_size=(128, 256, 256), overlap=(64, 64, 64)):
    model.eval()
    z, y, x = img.shape
    output_prob = np.zeros((3, z, y, x), dtype=np.float32)
    count_map = np.zeros((z, y, x), dtype=np.float32)

    sz, sy, sx = patch_size
    oz, oy, ox = overlap
    dz, dy, dx = sz - oz, sy - oy, sx - ox

    z_steps = range(0, max(1, z - oz), dz)
    y_steps = range(0, max(1, y - oy), dy)
    x_steps = range(0, max(1, x - ox), dx)
    
    with torch.no_grad():
        for zs in tqdm(z_steps, desc="UNet-BC Patching"):
            for ys in y_steps:
                for xs in x_steps:

                    ze, ye, xe = min(zs + sz, z), min(ys + sy, y), min(xs + sx, x)
                    zs_real, ys_real, xs_real = max(0, ze - sz), max(0, ye - sy), max(0, xe - sx)
                    
                    patch = img[zs_real:ze, ys_real:ye, xs_real:xe]
                    

                    ph, pw, pd = patch.shape
                    target_z = ((ph - 1) // 16 + 1) * 16
                    target_y = ((pw - 1) // 16 + 1) * 16
                    target_x = ((pd - 1) // 16 + 1) * 16
                    
                    pad_z = target_z - ph
                    pad_y = target_y - pw
                    pad_x = target_x - pd
                    
                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        patch = np.pad(patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant')

                    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    

                    logits = model(patch_t)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    

                    output_prob[:, zs_real:ze, ys_real:ye, xs_real:xe] += probs[:, :ph, :pw, :pd]
                    count_map[zs_real:ze, ys_real:ye, xs_real:xe] += 1

    return output_prob / np.maximum(count_map, 1e-7)

def post_process_bc(prob_map, core_thresh=0.5, boundary_thresh=0.5, min_size=20):
    
    core_prob = prob_map[1]
    boundary_prob = prob_map[2]
    

    seeds_mask = core_prob > core_thresh
    markers, n = nd_label(seeds_mask)
    counts = np.bincount(markers.ravel())
    mask_sizes = counts[markers] > min_size
    markers[~mask_sizes] = 0
    markers, _ = nd_label(markers > 0)
    
    mask = (core_prob + boundary_prob) > boundary_thresh
    labels = watershed(-core_prob, markers, mask=mask)
    return labels.astype(np.uint32)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    img_path = "./data/img/sample.tiff"
    seg_path = "./data/seg/sample.tiff"
    offsets_json = "./data/slice_offsets.json"
    output_dir = "./results/unet"
    model_path = './checkpoints/unet/best_model_bc.pth'

    print("Loading data and model...")
    img = tiff.imread(img_path)
    gt = tiff.imread(seg_path)
    original_shape = img.shape

    model = UNet3D_BC(in_channels=1, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
        print(f"Aligned shape: {aligned_img.shape}")
    else:
        print("Warning: No offsets found.")
        aligned_img, aligned_gt, shifts = img.astype(np.float32), gt, None

    slices = get_bbox_with_padding(aligned_gt, padding=32, shape_limit=aligned_img.shape)
    if slices is None: 
        print("GT is empty, skipping.")
        return

    img_crop = aligned_img[slices]
    gt_crop = aligned_gt[slices]

    print("UNet-BC Predicting...")
    img_norm = manual_normalize(img_crop)

    prob_map_crop = predict_large_image(model, img_norm, device, patch_size=(128, 256, 256))

    print("Watershed post-processing...")
    pred_labels_crop = post_process_bc(prob_map_crop)

    pred_labels_crop[gt_crop == 0] = 0

    full_aligned_pred = np.zeros(aligned_gt.shape, dtype=np.uint32)
    full_aligned_pred[slices] = pred_labels_crop

    print("Restoring to original coordinate system...")
    if shifts is not None:
        final_restored_seg = restore_original_segmentation(full_aligned_pred, shifts, original_shape)
    else:
        final_restored_seg = full_aligned_pred
    final_restored_seg = full_aligned_pred

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{fid}_unetbc_final.tiff")
    tiff.imwrite(save_path, final_restored_seg.astype(np.uint16), compression='zlib')

    print(f"✅ Success! Saved to: {save_path}")

if __name__ == '__main__':
    main()