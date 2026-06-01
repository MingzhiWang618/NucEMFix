import os
import json
import numpy as np
from skimage import io
import tifffile
from .realignment import apply_shift_with_padding, apply_shift_with_padding_seg

def load_offsets(json_path):
    
    with open(json_path, 'r') as f:
        return json.load(f)

def save_shifts(shifts, file_path, original_shape):
    

    serializable_shifts = [(float(dy), float(dx)) for dy, dx in shifts]
    
    with open(file_path, 'w') as f:
        json.dump({
            "shifts": serializable_shifts,
            "original_shape": list(original_shape)
        }, f)

def load_shifts(file_path):
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    shifts = [(dy, dx) for dy, dx in data["shifts"]]
    original_shape = tuple(data["original_shape"])
    return shifts, original_shape

def restore_original_image(aligned_volume, shifts, original_shape):
    """
    
    Args:
    
    Returns:
    """

    all_shifts = np.array(shifts)
    min_y, min_x = np.floor(np.min(all_shifts, axis=0)).astype(int)
    pad_top = max(0, -min_y)
    pad_left = max(0, -min_x)
    

    restored_volume = np.zeros(original_shape, dtype=aligned_volume.dtype)
    

    for z in range(original_shape[0]):
        dy, dx = shifts[z]
        dy_int = int(np.round(dy))
        dx_int = int(np.round(dx))
        

        aligned_slice = aligned_volume[z]
        

        y_start = pad_top + dy_int
        y_end = y_start + original_shape[1]
        x_start = pad_left + dx_int
        x_end = x_start + original_shape[2]
        

        y_start = max(0, min(y_start, aligned_slice.shape[0] - 1))
        y_end = max(0, min(y_end, aligned_slice.shape[0]))
        x_start = max(0, min(x_start, aligned_slice.shape[1] - 1))
        x_end = max(0, min(x_end, aligned_slice.shape[1]))
        

        if y_end > y_start and x_end > x_start:
            original_region = aligned_slice[y_start:y_end, x_start:x_end]
            restored_volume[z] = original_region
    
    return restored_volume

def restore_original_segmentation(aligned_seg, shifts, original_shape):
    """
    
    Args:
    
    Returns:
    """

    all_shifts = np.array(shifts)
    min_y, min_x = np.floor(np.min(all_shifts, axis=0)).astype(int)
    pad_top = max(0, -min_y)
    pad_left = max(0, -min_x)
    

    restored_seg = np.zeros(original_shape, dtype=aligned_seg.dtype)
    

    for z in range(original_shape[0]):
        dy, dx = shifts[z]
        dy_int = int(np.round(dy))
        dx_int = int(np.round(dx))
        

        aligned_slice = aligned_seg[z]
        

        y_start = pad_top + dy_int
        y_end = y_start + original_shape[1]
        x_start = pad_left + dx_int
        x_end = x_start + original_shape[2]
        

        y_start = max(0, min(y_start, aligned_slice.shape[0] - 1))
        y_end = max(0, min(y_end, aligned_slice.shape[0]))
        x_start = max(0, min(x_start, aligned_slice.shape[1] - 1))
        x_end = max(0, min(x_end, aligned_slice.shape[1]))
        

        if y_end > y_start and x_end > x_start:
            original_region = aligned_slice[y_start:y_end, x_start:x_end]
            restored_seg[z] = original_region
    
    return restored_seg

def process_single_file(img_path, seg_path, offsets_json, output_dir):
    """
    
    Args:
    """

    os.makedirs(output_dir, exist_ok=True)
    

    file_id = os.path.basename(img_path).split('_')[0]
    

    offsets = load_offsets(offsets_json)
    if file_id not in offsets:
        raise ValueError(f"No offset information found for file {file_id}")
    

    img = io.imread(img_path)
    seg = io.imread(seg_path)
    original_shape = img.shape
    

    print(f"Original image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Original segmentation shape: {seg.shape}, dtype: {seg.dtype}")
    

    offset_info = offsets[str(file_id)]
    relative_shifts = offset_info['relative_shifts']
    bad_slices = offset_info['bad_slices']
    

    flow_volume = []
    for shift_info in relative_shifts:
        if shift_info['status'] == 'skip':
            flow_volume.append({'skip': True})
        else:
            flow_volume.append({
                'dx': shift_info['dx'],
                'dy': shift_info['dy'],
                'quality': shift_info['quality'],
                'ref_index': shift_info['from_slice'],
                'mov_index': shift_info['to_slice']
            })
    

    aligned_img, shifts, valid_indices = apply_shift_with_padding(img, flow_volume, bad_slices)
    aligned_seg = apply_shift_with_padding_seg(seg, shifts)
    

    aligned_img_path = os.path.join(output_dir, f'{file_id}_aligned.tif')
    aligned_seg_path = os.path.join(output_dir, f'{file_id}_aligned_seg.tif')
    
    tifffile.imwrite(aligned_img_path, aligned_img, compression='zlib')
    tifffile.imwrite(aligned_seg_path, aligned_seg, compression='zlib')
    

    print(f"Aligned image shape: {aligned_img.shape}, dtype: {aligned_img.dtype}")
    print(f"Aligned segmentation shape: {aligned_seg.shape}, dtype: {aligned_seg.dtype}")
    

    shifts_path = os.path.join(output_dir, f'{file_id}_shifts.json')
    save_shifts(shifts, shifts_path, original_shape)
    

    aligned_img_restore = io.imread(aligned_img_path)
    aligned_seg_restore = io.imread(aligned_seg_path)
    

    shifts_restore, original_shape_restore = load_shifts(shifts_path)
    

    restored_img = restore_original_image(aligned_img_restore, shifts_restore, original_shape_restore)
    restored_seg = restore_original_segmentation(aligned_seg_restore, shifts_restore, original_shape_restore)
    

    print(f"Restored image shape: {restored_img.shape}, dtype: {restored_img.dtype}")
    print(f"Restored segmentation shape: {restored_seg.shape}, dtype: {restored_seg.dtype}")
    

    restored_img_path = os.path.join(output_dir, f'{file_id}_restored.tif')
    restored_seg_path = os.path.join(output_dir, f'{file_id}_restored_seg.tif')
    
    tifffile.imwrite(restored_img_path, restored_img, compression='zlib')
    tifffile.imwrite(restored_seg_path, restored_seg, compression='zlib')
    
    return {
        'file_id': file_id,
        'aligned_img': aligned_img_path,
        'aligned_seg': aligned_seg_path,
        'restored_img': restored_img_path,
        'restored_seg': restored_seg_path,
        'shifts': shifts_path
    }

if __name__ == "__main__":

    img_path = r"D:\paper\fafb_process\tmp\8533661_img.tiff"
    seg_path = r"D:\paper\fafb_process\tmp\8533661_mis.tiff"
    offsets_json = r"D:\paper\fafb_data\merge_error\merge_error_correct_6.14\slice_offsets.json"
    output_dir = r"D:\paper\fafb_process\tmp\aligned"
    

    try:
        result = process_single_file(img_path, seg_path, offsets_json, output_dir)
        print("\nProcessing completed successfully!")
        print(f"File ID: {result['file_id']}")
        print(f"Aligned image: {result['aligned_img']}")
        print(f"Aligned segmentation: {result['aligned_seg']}")
        print(f"Restored image: {result['restored_img']}")
        print(f"Restored segmentation: {result['restored_seg']}")
        print(f"Shifts saved to: {result['shifts']}")
    except Exception as e:
        print(f"Error: {str(e)}")