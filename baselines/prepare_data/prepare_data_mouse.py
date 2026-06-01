import numpy as np
import h5py
import tifffile
import os
from pathlib import Path
from tqdm import tqdm
import random

def calculate_padded_bbox(bbox, target_size=96):
    """
    """
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    

    z_center = (z_min + z_max) // 2
    y_center = (y_min + y_max) // 2
    x_center = (x_min + x_max) // 2
    

    half_size = target_size // 2
    

    new_z_start = z_center - half_size
    new_z_end = new_z_start + target_size
    
    new_y_start = y_center - half_size
    new_y_end = new_y_start + target_size
    
    new_x_start = x_center - half_size
    new_x_end = new_x_start + target_size
    
    return [new_z_start, new_z_end, new_y_start, new_y_end, new_x_start, new_x_end]

def crop_and_pad_data(data_source, padded_bbox, target_size, fill_value=0):
    """
    """
    z_s, z_e, y_s, y_e, x_s, x_e = padded_bbox
    full_shape = data_source.shape
    

    output = np.full((target_size, target_size, target_size), fill_value, dtype=data_source.dtype)
    

    z_s_valid = max(0, z_s)
    z_e_valid = min(full_shape[0], z_e)
    y_s_valid = max(0, y_s)
    y_e_valid = min(full_shape[1], y_e)
    x_s_valid = max(0, x_s)
    x_e_valid = min(full_shape[2], x_e)
    

    if z_s_valid < z_e_valid and y_s_valid < y_e_valid and x_s_valid < x_e_valid:

        out_z_s = z_s_valid - z_s
        out_z_e = out_z_s + (z_e_valid - z_s_valid)
        out_y_s = y_s_valid - y_s
        out_y_e = out_y_s + (y_e_valid - y_s_valid)
        out_x_s = x_s_valid - x_s
        out_x_e = out_x_s + (x_e_valid - x_s_valid)
        

        output[out_z_s:out_z_e, out_y_s:out_y_e, out_x_s:out_x_e] = data_source[z_s_valid:z_e_valid, y_s_valid:y_e_valid, x_s_valid:x_e_valid]
        
    return output

def load_segment_bbox(seg_id, seg_txt, seg_h5_fid):
    """
    """

    matched = seg_txt[seg_txt[:, 0] == seg_id]
    if len(matched) == 0:
        raise ValueError(f"ID {seg_id} not found in txt file.")
    
    seg = matched[0]
    

    z_start, z_end = int(seg[1]), int(seg[2])
    y_start, y_end = int(seg[3]), int(seg[4])
    x_start, x_end = int(seg[5]), int(seg[6])
    
    bbox = [z_start, z_end, y_start, y_end, x_start, x_end]
    return bbox

def process_single_id(seg_id, seg_h5_fid, img_h5_fid, seg_txt, output_seg_dir, output_img_dir, target_size=96):
    """
    """
    try:

        bbox = load_segment_bbox(seg_id, seg_txt, seg_h5_fid)
        

        padded_bbox = calculate_padded_bbox(bbox, target_size)
        

        img_cropped = crop_and_pad_data(img_h5_fid, padded_bbox, target_size, fill_value=0)
        

        seg_padded = crop_and_pad_data(seg_h5_fid, padded_bbox, target_size, fill_value=0)
       

        seg_output_path = os.path.join(output_seg_dir, f"{seg_id}.tiff")
        img_output_path = os.path.join(output_img_dir, f"{seg_id}_img.tiff")
        

        tifffile.imwrite(seg_output_path, seg_padded.astype(np.uint8)) 
        tifffile.imwrite(img_output_path, img_cropped.astype(np.uint8))
        
        return True
    except Exception as e:
        print(f"")
        return False

def main():

    normal_nuclei_file = "./data"
    seg_h5_file = "./data"
    img_h5_file = "./data"
    seg_txt_file = "./data"
    
    output_seg_dir = "./data"
    output_img_dir = "./data"
    
    os.makedirs(output_seg_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)
    

    with open(normal_nuclei_file, 'r') as f:
        id_list = [int(line.strip()) for line in f if line.strip()]
    

    if len(id_list) > 1000:

        id_list = id_list[:1000]
    
    print(f"")
    

    seg_txt = np.loadtxt(seg_txt_file)
    
    with h5py.File(seg_h5_file, 'r') as seg_f, h5py.File(img_h5_file, 'r') as img_f:

        seg_h5_fid = seg_f['main']
        img_h5_fid = img_f['main']
        
        success_count = 0
        for seg_id in tqdm(id_list, desc="Processing"):
            success = process_single_id(seg_id, seg_h5_fid, img_h5_fid, seg_txt, 
                                      output_seg_dir, output_img_dir, target_size=96)
            if success:
                success_count += 1
    
    print(f"")

if __name__ == "__main__":
    main()