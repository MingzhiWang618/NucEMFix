import numpy as np
import h5py
import os
import tifffile
from tqdm import tqdm

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

def remap_ids_to_uint8(seg_array):
    """
    """
    unique_ids = np.unique(seg_array)
    unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) > 255:
        print(f"")
    
    new_seg = np.zeros_like(seg_array, dtype=np.uint8)

    for i, old_id in enumerate(unique_ids[:255]):
        new_seg[seg_array == old_id] = i + 1
    return new_seg

def main():

    seg_norm_txt = "./data"
    seg_h5_path = r"./data"
    seg_txt_path = r"./data"
    

    local_img_dir = "./data"
    

    output_img_dir = r"./data"
    output_seg_dir = r"./data"
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_seg_dir, exist_ok=True)

    seg_txt = np.loadtxt(seg_txt_path, dtype=int)
    seg_norm = np.loadtxt(seg_norm_txt, dtype=int)
    target_ids = seg_norm[:1000]
    target_shape = [128, 128, 128] # [Z, Y, X]

    with h5py.File(seg_h5_path, 'r') as f:
        seg_main = f['main']
        full_shape = seg_main.shape

        for seg_id in tqdm(target_ids, desc="Processing Local FAFB Data"):
            try:

                img_name = f"{seg_id}.tiff"
                src_img_path = os.path.join(local_img_dir, img_name)
                
                if not os.path.exists(src_img_path):

                    print(f"")
                    continue

                bbox= load_segment_bbox(seg_id, seg_txt, seg_main, )
                c_z = (bbox[0] + bbox[1]) // 2
                c_y = (bbox[2] + bbox[3]) // 2
                c_x = (bbox[4] + bbox[5]) // 2

                z_range = [c_z - target_shape[0]//2, c_z + target_shape[0]//2]
                y_range = [c_y - target_shape[1]//2, c_y + target_shape[1]//2]
                x_range = [c_x - target_shape[2]//2, c_x + target_shape[2]//2]

                zs, ze = max(0, z_range[0]), min(full_shape[0], z_range[1])
                ys, ye = max(0, y_range[0]), min(full_shape[1], y_range[1])
                xs, xe = max(0, x_range[0]), min(full_shape[2], x_range[1])

                seg_crop_raw = np.zeros(target_shape, dtype=seg_main.dtype)
                
                out_zs = zs - z_range[0]
                out_ys = ys - y_range[0]
                out_xs = xs - x_range[0]

                raw_data = seg_main[zs:ze, ys:ye, xs:xe]
                seg_crop_raw[out_zs:out_zs+raw_data.shape[0],
                             out_ys:out_ys+raw_data.shape[1],
                             out_xs:out_xs+raw_data.shape[2]] = raw_data

                seg_crop_final = remap_ids_to_uint8(seg_crop_raw)

                img_data = tifffile.imread(src_img_path)
                

                tifffile.imwrite(os.path.join(output_img_dir, img_name), img_data)
                tifffile.imwrite(os.path.join(output_seg_dir, f"{seg_id}_seg.tiff"), seg_crop_final)

            except Exception as e:
                print(f"")
                continue

if __name__ == "__main__":
    main()