import os
import re
import sys
import torch
import numpy as np
import tifffile as tiff
from tqdm import tqdm

from monai.transforms import (
    Compose, 
    EnsureChannelFirst,
    NormalizeIntensity, 
    ToTensor
)
from monai.networks.nets import DynUNet
from monai.inferers import sliding_window_inference
from skimage.segmentation import watershed
from scipy.ndimage import label as nd_label

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

def post_process_bc(prob_map, core_thresh=0.5, boundary_thresh=0.5, min_size=20):
    
    core_prob = prob_map[1]
    boundary_prob = prob_map[2]
    

    seeds_mask = core_prob > core_thresh
    markers, n = nd_label(seeds_mask)
    if n > 0:
        counts = np.bincount(markers.ravel())
        mask_sizes = counts[markers] > min_size
        markers[~mask_sizes] = 0
        markers, _ = nd_label(markers > 0)
    
    mask = (core_prob + boundary_prob) > boundary_thresh
    labels = watershed(-core_prob, markers, mask=mask)
    return labels.astype(np.uint32)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    img_path = "./data/img/sample.tiff"
    seg_path = "./data/seg/sample.tiff"
    offsets_json = "./data/slice_offsets.json"
    output_dir = "./results/nnunet"
    model_path = './checkpoints/nnunet/best_model.pth'

    print("Loading data...")
    img = tiff.imread(img_path)
    gt = tiff.imread(seg_path)
    original_shape = img.shape

    print("Loading nnUNet model...")
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        kernel_size=[[3,3,3]]*5,
        strides=[[1,1,1]] + [[2,2,2]]*4,
        upsample_kernel_size=[[2,2,2]]*4,
        norm_name="instance",
        deep_supervision=False,
    ).to(device)
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
        aligned_img, aligned_gt, shifts = img.astype(np.float32), gt, None

    slices = get_bbox_with_padding(aligned_gt, padding=32, shape_limit=aligned_img.shape)
    if slices is None: 
        print("GT is empty, skipping.")
        return

    img_crop = aligned_img[slices]
    gt_crop = aligned_gt[slices]

    pre_transforms = Compose([
        EnsureChannelFirst(channel_dim="no_channel"), 
        NormalizeIntensity(),
        ToTensor()
    ])
    input_tensor = pre_transforms(img_crop).unsqueeze(0).to(device) # [1, 1, Z, Y, X]

    print("Sliding window inference...")
    roi_size = (128, 128, 128) 
    sw_batch_size = 4 
    
    with torch.no_grad():
        output_logits = sliding_window_inference(
            inputs=input_tensor, 
            roi_size=roi_size, 
            sw_batch_size=sw_batch_size, 
            predictor=model,
            overlap=0.25,
            mode="gaussian"
        )

        prob_map_crop = torch.softmax(output_logits, dim=1).cpu().numpy()[0]

    print("Watershed post-processing...")
    pred_labels_crop = post_process_bc(prob_map_crop)

    pred_labels_crop[gt_crop == 0] = 0

    full_aligned_pred = np.zeros(aligned_gt.shape, dtype=np.uint32)
    full_aligned_pred[slices] = pred_labels_crop
    shifts = None 
    print("Restoring to original coordinate system...")
    if shifts is not None:
        final_restored_seg = restore_original_segmentation(full_aligned_pred, shifts, original_shape)
    else:
        final_restored_seg = full_aligned_pred

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{fid}_nnunet_bc_final.tiff")
    print(f"Unique labels after BC: {np.unique(final_restored_seg)}")
    tiff.imwrite(save_path, final_restored_seg.astype(np.uint16), compression='zlib')

    print(f"✅ Success! Saved to: {save_path}")

if __name__ == '__main__':
    main()