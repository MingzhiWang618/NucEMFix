import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import binary_dilation, shift, gaussian_filter
from skimage import io, exposure, filters
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.ndimage import shift as ndi_shift
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.util import view_as_windows
import cv2
import matplotlib.patches as patches

def enhance_contrast(img, clip_limit=0.03):
    
    return exposure.equalize_adapthist(img, clip_limit=clip_limit)

def preprocess_image(img, sigma=1.0):
    
    img = enhance_contrast(img)
    return gaussian_filter(img, sigma=sigma)

def detect_bad_slice(img, black_ratio_thresh=0.3):
    """
    """

    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    

    black_ratio = np.mean(img < 0.1)
    
    if black_ratio > black_ratio_thresh:
        return True
    return False

def phase_cross_correlation(img1, img2, upsample_factor=1):
    """
    """

    f1 = fft2(img1)
    f2 = fft2(img2)
    

    cross_power_spectrum = (f1 * np.conj(f2)) / (np.abs(f1) * np.abs(f2) + 1e-6)
    cross_correlation = np.real(fftshift(ifft2(cross_power_spectrum)))
    

    max_loc = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    shift_y = max_loc[0] - img1.shape[0] // 2
    shift_x = max_loc[1] - img1.shape[1] // 2
    

    peak_value = cross_correlation[max_loc]
    mean_value = np.mean(cross_correlation)
    quality = peak_value / (mean_value + 1e-6)
    

    if upsample_factor > 1:

        region_size = 5
        start_y = max(0, max_loc[0] - region_size // 2)
        end_y = min(img1.shape[0], max_loc[0] + region_size // 2 + 1)
        start_x = max(0, max_loc[1] - region_size // 2)
        end_x = min(img1.shape[1], max_loc[1] + region_size // 2 + 1)
        
        region = cross_correlation[start_y:end_y, start_x:end_x]
        upsampled = cv2.resize(region, None, fx=upsample_factor, fy=upsample_factor, 
                                 interpolation=cv2.INTER_CUBIC)
        
        max_loc_upsampled = np.unravel_index(np.argmax(upsampled), upsampled.shape)
        

        shift_y += (max_loc_upsampled[0] - region_size * upsample_factor // 2) / upsample_factor
        shift_x += (max_loc_upsampled[1] - region_size * upsample_factor // 2) / upsample_factor
    
    return shift_x, shift_y, quality

def batch_cross_correlation_fft(patches1, patches2):
    """

    :param patches1: shape = (N, H, W)
    :param patches2: shape = (N, H, W)
    :return: dx, dy, quality (per patch)
    """
    assert patches1.shape == patches2.shape
    N, H, W = patches1.shape

    mean1 = np.mean(patches1, axis=(1, 2), keepdims=True)
    mean2 = np.mean(patches2, axis=(1, 2), keepdims=True)
    std1 = np.std(patches1, axis=(1, 2), keepdims=True)
    std2 = np.std(patches2, axis=(1, 2), keepdims=True)

    valid = (std1[:, 0, 0] > 1e-3) & (std2[:, 0, 0] > 1e-3)

    norm1 = patches1 - mean1
    norm2 = patches2 - mean2

    fft1 = fft2(norm1, axes=(1, 2))
    fft2_conj = np.conj(fft2(norm2, axes=(1, 2)))
    cross_corr = np.real(ifft2(fft1 * fft2_conj, axes=(1, 2)))

    cross_corr_shifted = fftshift(cross_corr, axes=(1, 2))

    peak_idx = np.argmax(cross_corr_shifted.reshape(N, -1), axis=1)
    peak_y = (peak_idx // W) - (H // 2)
    peak_x = (peak_idx % W) - (W // 2)

    denom = (std1 * std2 * H * W + 1e-6).reshape(N)

    peak_vals = cross_corr_shifted.reshape(N, -1)[np.arange(N), peak_idx]

    quality = np.where(valid, peak_vals / denom, 0)
    dx = np.where(valid, peak_x, 0)
    dy = np.where(valid, peak_y, 0)

    return dx, dy, quality

def check_patch_black_ratio(patch, black_ratio_thresh=0.3):
    """
    """

    patch = patch.astype(np.float32)
    if patch.max() > 1:
        patch = patch / 255.0
    

    black_ratio = np.mean(patch < 0.1)
    return black_ratio <= black_ratio_thresh

def local_cross_correlation_robust(ref_img, mov_img, patch_size=128, stride=64, max_shift=50, quality_thresh=0.1):
    """
    """

    ref_img_proc = preprocess_image(ref_img)
    mov_img_proc = preprocess_image(mov_img)
    

    window_shape = (patch_size, patch_size)
    
    # Calculate possible starting positions for patches
    ref_rows = (ref_img_proc.shape[0] - patch_size) // stride + 1
    ref_cols = (ref_img_proc.shape[1] - patch_size) // stride + 1

    patches_ref = view_as_windows(ref_img_proc, window_shape, step=stride).reshape(-1, patch_size, patch_size)
    patches_mov = view_as_windows(mov_img_proc, window_shape, step=stride).reshape(-1, patch_size, patch_size)

    n_patches = patches_ref.shape[0]
    
    # Store original patch indices to map back later
    patch_origins = []
    for r_idx in range(ref_rows):
        for c_idx in range(ref_cols):
            patch_origins.append((r_idx * stride, c_idx * stride))
    patch_origins = np.array(patch_origins)

    valid_patches_mask = np.array([check_patch_black_ratio(p) for p in patches_ref]) & \
                           np.array([check_patch_black_ratio(p) for p in patches_mov])
    
    if np.sum(valid_patches_mask) < 5:
        return None, None, None, None, None # Also return best patch info
    

    valid_ref = patches_ref[valid_patches_mask]
    valid_mov = patches_mov[valid_patches_mask]
    

    dx_batch, dy_batch, q_batch = batch_cross_correlation_fft(valid_ref, valid_mov)
    

    valid_match_mask = (q_batch > quality_thresh) & \
                       (np.abs(dx_batch) < max_shift) & \
                       (np.abs(dy_batch) < max_shift)
    
    if np.sum(valid_match_mask) < 5:
        return None, None, None, None, None # Also return best patch info
    
    # Get the original indices of the valid and well-matched patches
    overall_valid_indices = np.where(valid_patches_mask)[0][valid_match_mask]

    # Find the best matched patch (highest quality) among the valid ones
    best_match_idx_in_valid_matched = np.argmax(q_batch[valid_match_mask])
    best_overall_patch_idx = overall_valid_indices[best_match_idx_in_valid_matched]

    best_patch_ref_origin = patch_origins[best_overall_patch_idx]
    best_patch_dx = dx_batch[valid_match_mask][best_match_idx_in_valid_matched]
    best_patch_dy = dy_batch[valid_match_mask][best_match_idx_in_valid_matched]

    mean_dx = np.mean(dx_batch[valid_match_mask])
    mean_dy = np.mean(dy_batch[valid_match_mask])
    mean_quality = np.mean(q_batch[valid_match_mask])
    
    return mean_dx, mean_dy, mean_quality, best_patch_ref_origin, (best_patch_dx, best_patch_dy)

def compute_displacement_volume(volume, patch_size=128, stride=64, max_shift=50, min_shift=2):
    
    flow_volume = []
    bad_slices = []
    
    # Detect all bad slices (using original image)
    for z in range(len(volume)):
        if detect_bad_slice(volume[z]):
            bad_slices.append(z)
    
    # Find the first good slice to use as the initial reference
    prev_good_index = -1
    for z in range(len(volume)):
        if z not in bad_slices:
            prev_good_index = z
            break
    
    if prev_good_index == -1:
        print("❗ Warning: All slices are bad. Cannot perform alignment.")
        # If all slices are bad, populate flow_volume with skip flags
        for _ in range(len(volume) - 1):
            flow_volume.append({'skip': True})
        return flow_volume, bad_slices
        
    # Preprocess the entire volume (for alignment calculations)
    proc_volume = np.array([preprocess_image(img) for img in volume])
    
    # Calculate displacements
    for z in tqdm(range(len(volume) - 1), desc="Computing displacement fields"):
        current_slice_index = z + 1 # The moving slice in this pair (compared to prev_good_index)

        # If the current moving slice is bad, or the current reference slice is bad, skip calculation for this pair.
        # We will infer its shift later.
        if current_slice_index in bad_slices:
            flow_volume.append({'skip': True})
            continue # Move to the next slice

        # If prev_good_index itself is a bad slice (shouldn't happen with the logic above, but as a safeguard)
        if prev_good_index in bad_slices:
             # This means we don't have a valid previous good slice, which is an error in logic or extreme case
             print(f"Error: prev_good_index {prev_good_index} is a bad slice, which should not be used as reference.")
             flow_volume.append({'skip': True})
             continue

        # Proceed only if both the reference and the current moving slice are good
        dx, dy, quality, _, _ = local_cross_correlation_robust(
            proc_volume[prev_good_index], 
            proc_volume[current_slice_index], # Use current_slice_index as the moving slice
            patch_size=patch_size,
            stride=stride,
            max_shift=max_shift
        )
        
        if dx is None or np.hypot(dx, dy) < min_shift:
            # If no reliable shift found, treat as no significant shift or a failed match
            dx, dy, quality = 0, 0, 0
            # Note: We still record this as a valid calculation attempt, even if shift is 0
            # If it's truly problematic, 'quality' will be low.
        
        flow_volume.append({
            'dx': dx,
            'dy': dy,
            'quality': quality,
            'ref_index': prev_good_index,
            'mov_index': current_slice_index # Record the actual moving slice index
        })
        
        # IMPORTANT: Update prev_good_index ONLY if the *current_slice_index* was good
        # and a successful calculation was made.
        # If the current_slice_index was skipped, prev_good_index remains the same.
        prev_good_index = current_slice_index 
    
    return flow_volume, bad_slices

def compute_average_shift(result):
    
    if 'skip' in result:
        return 0, 0
    return result['dx'], result['dy']

def apply_shift_with_padding(volume, flow_volume, bad_slices):
    """
    """

    shifts = [(0, 0)]
    
    # Ensure shifts list has an entry for every slice
    # Initialize with default (0,0) for all slices, then fill with actual shifts
    for _ in range(len(volume) - 1):
        shifts.append((0,0)) # Placeholder for each subsequent slice

    # Find the first good slice to base the alignment
    first_good_idx = -1
    for i in range(len(volume)):
        if i not in bad_slices:
            first_good_idx = i
            break
    
    if first_good_idx == -1: # All slices are bad
        print("Warning: All slices are bad, no alignment performed. Returning original shifts (0,0).")
        return volume, shifts, [] # Return original volume, all zero shifts, and empty valid_indices

    # Set the initial reference shift for the first good slice
    cumulative_shifts = np.zeros((len(volume), 2), dtype=float)
    cumulative_shifts[first_good_idx] = (0, 0) # The first good slice is the reference point for its group

    last_valid_cumulative_shift = (0,0) # Stores the cumulative shift of the *last successfully processed* good slice.
    last_valid_slice_idx = 0 # Stores the index of the *last successfully processed* good slice.

    # Propagate shifts forward
    for i in range(len(volume) - 1): # flow_volume has len(volume)-1 entries
        current_flow_info = flow_volume[i]
        target_slice_idx = i + 1 # The slice this flow_info is *for* (mov_index)

        if 'skip' in current_flow_info:
            # If the current target slice or its determined reference was skipped,
            # its cumulative shift is the same as the last successfully aligned slice.
            cumulative_shifts[target_slice_idx] = last_valid_cumulative_shift
        else:
            # This flow_info represents a valid shift calculation
            ref_idx = current_flow_info['ref_index']
            dx, dy = current_flow_info['dx'], current_flow_info['dy']
            
            # The shift (dx, dy) is the displacement of 'mov_index' relative to 'ref_index'.
            # To get cumulative shift of 'mov_index', add its relative shift to cumulative shift of 'ref_index'.
            cumulative_shifts[target_slice_idx] = (cumulative_shifts[ref_idx][0] + dy, cumulative_shifts[ref_idx][1] + dx)
            
            last_valid_cumulative_shift = cumulative_shifts[target_slice_idx]
            last_valid_slice_idx = target_slice_idx
    
    # Now, fill in any leading bad slices or gaps where `last_valid_cumulative_shift` was not updated
    current_inferred_shift = (0,0)
    for i in range(len(volume)):
        if i not in bad_slices: # If it's a good slice, its shift is already computed or is 0
            current_inferred_shift = cumulative_shifts[i]
        else: # If it's a bad slice, it inherits the shift of the last good slice.
            cumulative_shifts[i] = current_inferred_shift
    
    # Make sure the shifts list contains tuples (dy, dx) matching the expected output.
    shifts = [tuple(s) for s in cumulative_shifts]

    all_shifts_np = np.array(shifts)
    min_y, min_x = np.floor(np.min(all_shifts_np, axis=0)).astype(int)
    max_y, max_x = np.ceil(np.max(all_shifts_np, axis=0)).astype(int)

    pad_top = max(0, -min_y)
    pad_bottom = max(0, max_y)
    pad_left = max(0, -min_x)
    pad_right = max(0, max_x)

    base_h, base_w = volume.shape[1:]
    new_h = base_h + pad_top + pad_bottom
    new_w = base_w + pad_left + pad_right

    aligned_volume = np.zeros((volume.shape[0], new_h, new_w), dtype=volume.dtype)
    

    for z in range(volume.shape[0]):
        dy, dx = shifts[z]
        img = volume[z]
        

        dy_int = int(np.round(dy))
        dx_int = int(np.round(dx))
        

        temp_canvas = np.zeros((new_h, new_w), dtype=volume.dtype)

        temp_canvas[pad_top:pad_top+base_h, pad_left:pad_left+base_w] = img
        

        if dy_int != 0 or dx_int != 0:

            src_y_start = max(0, -dy_int)
            src_y_end = min(new_h, new_h - dy_int)
            src_x_start = max(0, -dx_int)
            src_x_end = min(new_w, new_w - dx_int)
            

            dst_y_start = max(0, dy_int)
            dst_y_end = min(new_h, new_h + dy_int)
            dst_x_start = max(0, dx_int)
            dst_x_end = min(new_w, new_w + dx_int)
            

            temp_canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                temp_canvas[src_y_start:src_y_end, src_x_start:src_x_end]
            

            if dy_int > 0:
                temp_canvas[:dy_int, :] = 0
            elif dy_int < 0:
                temp_canvas[dy_int:, :] = 0
            if dx_int > 0:
                temp_canvas[:, :dx_int] = 0
            elif dx_int < 0:
                temp_canvas[:, dx_int:] = 0
        
        aligned_volume[z] = temp_canvas

    # Return valid_indices for visualization, which are the good slices
    valid_indices = [i for i in range(len(volume)) if i not in bad_slices]

    return aligned_volume, shifts, valid_indices

def apply_shift_with_padding_seg(volume, shifts):
    """
    """

    all_shifts_np = np.array(shifts)
    min_y, min_x = np.floor(np.min(all_shifts_np, axis=0)).astype(int)
    max_y, max_x = np.ceil(np.max(all_shifts_np, axis=0)).astype(int)

    pad_top = max(0, -min_y)
    pad_bottom = max(0, max_y)
    pad_left = max(0, -min_x)
    pad_right = max(0, max_x)

    base_h, base_w = volume.shape[1:]
    new_h = base_h + pad_top + pad_bottom
    new_w = base_w + pad_left + pad_right

    aligned_volume = np.zeros((volume.shape[0], new_h, new_w), dtype=volume.dtype)
    

    for z in range(volume.shape[0]):
        dy, dx = shifts[z]
        img = volume[z]
        

        dy_int = int(np.round(dy))
        dx_int = int(np.round(dx))
        

        temp_canvas = np.zeros((new_h, new_w), dtype=volume.dtype)

        temp_canvas[pad_top:pad_top+base_h, pad_left:pad_left+base_w] = img
        

        if dy_int != 0 or dx_int != 0:

            src_y_start = max(0, -dy_int)
            src_y_end = min(new_h, new_h - dy_int)
            src_x_start = max(0, -dx_int)
            src_x_end = min(new_w, new_w - dx_int)
            

            dst_y_start = max(0, dy_int)
            dst_y_end = min(new_h, new_h + dy_int)
            dst_x_start = max(0, dx_int)
            dst_x_end = min(new_w, new_w + dx_int)
            

            temp_canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                temp_canvas[src_y_start:src_y_end, src_x_start:src_x_end]
            

            if dy_int > 0:
                temp_canvas[:dy_int, :] = 0
            elif dy_int < 0:
                temp_canvas[dy_int:, :] = 0
            if dx_int > 0:
                temp_canvas[:, :dx_int] = 0
            elif dx_int < 0:
                temp_canvas[:, dx_int:] = 0
        
        aligned_volume[z] = temp_canvas

    return aligned_volume

def visualize_best_matches(ref_img, mov_img, patch_size=128, stride=64, max_shift=50, quality_thresh=0.1):
    """
    """
    # Get the overall shift and the best patch information
    mean_dx, mean_dy, mean_quality, best_patch_ref_origin, best_patch_relative_shift = \
        local_cross_correlation_robust(
            ref_img, mov_img, patch_size, stride, max_shift, quality_thresh
        )
    
    if mean_dx is None:
        print(f"Could not find a reliable match between slices.")
        return None

    # Unpack best patch information
    best_ref_y, best_ref_x = best_patch_ref_origin
    best_patch_dx, best_patch_dy = best_patch_relative_shift

    # Create visualization images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)) # Increased figure size

    # Display reference image and moving image
    ax1.imshow(ref_img, cmap='gray')
    ax2.imshow(mov_img, cmap='gray')
    
    # Draw rectangle around the best matched patch in the reference image
    rect_ref = patches.Rectangle((best_ref_x, best_ref_y), patch_size, patch_size,
                                 linewidth=2, edgecolor='r', facecolor='none', label='Best Match Patch')
    ax1.add_patch(rect_ref)
    
    # Calculate and draw rectangle around the corresponding patch in the moving image
    # The origin of the best patch in the moving image is its origin in ref + its relative shift
    mov_patch_origin_x = best_ref_x + best_patch_dx
    mov_patch_origin_y = best_ref_y + best_patch_dy

    rect_mov = patches.Rectangle((mov_patch_origin_x, mov_patch_origin_y), patch_size, patch_size,
                                 linewidth=2, edgecolor='r', facecolor='none')
    ax2.add_patch(rect_mov)

    # Add center markers (optional, but good for overall shift visualization)
    center_y, center_x = ref_img.shape[0] // 2, ref_img.shape[1] // 2
    ax1.plot(center_x, center_y, 'b+', markersize=10, label='Image Center')
    ax2.plot(center_x + mean_dx, center_y + mean_dy, 'b+', markersize=10, label='Shifted Center')

    # Add titles and displacement information
    ax1.set_title('Reference Slice (with Best Match Patch)')
    ax2.set_title(f'Moving Slice (Corresponding Patch)\nOverall Shift: dx={mean_dx:.2f}, dy={mean_dy:.2f}, quality={mean_quality:.3f}')
    
    ax1.legend()
    plt.tight_layout()
    return fig

if __name__ == "__main__":

    img = io.imread(r"D:\paper\fafb_process\tmp\1366097_img.tiff")
    seg = io.imread(r"D:\paper\fafb_process\tmp\1366097_ms_mis.tiff")
    

    flow_volume, bad_slices = compute_displacement_volume(
        img, 
        patch_size=256,
        stride=32,
        max_shift=200,
        min_shift=2,
    )
    

    aligned_img, shifts, valid_indices = apply_shift_with_padding(img, flow_volume, bad_slices)
    

    aligned_seg = apply_shift_with_padding_seg(seg, shifts)
    

    save_path_img = r"D:\paper\fafb_process\tmp\1366097_aligned.tif"      
    save_path_seg = r"D:\paper\fafb_process\tmp\1366097_aligned_seg.tif"           
    io.imsave(save_path_img, aligned_img)
    io.imsave(save_path_seg, aligned_seg)
    

    vis_folder = 'D:/paper/fafb_process/tmp/best_match_slice/'
    os.makedirs(vis_folder, exist_ok=True)
    

    # Ensure there are enough valid indices to compare adjacent slices
    if len(valid_indices) > 1:
        for i in range(len(valid_indices)-1):
            ref_idx = valid_indices[i]
            mov_idx = valid_indices[i+1]

            ref_slice = img[ref_idx]
            mov_slice = img[mov_idx]
            
            print(f"Visualizing match for slice pair: {ref_idx} to {mov_idx}")
            fig = visualize_best_matches(ref_slice, mov_slice, 
                                         patch_size=256, 
                                         stride=32,
                                         max_shift=200,
                                         quality_thresh=0.1)
            
            if fig is not None:
                plt.savefig(os.path.join(vis_folder, f'best_match_slice_{ref_idx}_to_{mov_idx}.png'))
                plt.close(fig)
            else:
                print(f"Skipping visualization for {ref_idx} to {mov_idx} due to no reliable match.")
    else:
        print("Not enough good slices to perform visualization of best matches between adjacent slices.")

    slice_shifts = []
    # Adjusting range for flow_volume as it contains n-1 entries for n slices
    for z in range(len(img) - 1): 
        flow = flow_volume[z]
        if 'skip' in flow:
            dx, dy = 0, 0
            quality = 0 # No quality for skipped
        else:
            dx, dy = flow['dx'], flow['dy']
            quality = flow['quality']
        
        slice_info = {
            "slice_pair": f"{z}->{z+1}",
            "dx": dx,
            "dy": dy,
            "magnitude": np.hypot(dx, dy),
            "quality": quality,
            "status": "✅ Good" if (z not in bad_slices and z+1 not in bad_slices) else "❗ Bad Pair"
        }
        slice_shifts.append(slice_info)
    

    print("\n" + "="*60)
    print("✅ Alignment complete and results saved.")
    print(f"✅ Saved aligned image to: {save_path_img}")
    print(f"✅ Saved aligned segmentation to: {save_path_seg}")
    print(f"❗ Total bad slices found: {len(bad_slices)}")
    print(f"❗ Bad slice indices: {bad_slices}")
    print(f"✅ Total good slices in the original volume: {len([s for s in range(len(img)) if s not in bad_slices])}")
    

    print("\n" + "="*60)
    print("Relative Displacement Details (between original adjacent slices):")
    print("Slice Pair | X-Shift | Y-Shift | Magnitude | Quality | Status")
    print("-"*70)
    
    for shift_info in slice_shifts:
        print(f"{shift_info['slice_pair']:<10s} | {shift_info['dx']:7.2f} | "
              f"{shift_info['dy']:7.2f} | {shift_info['magnitude']:9.2f} | {shift_info['quality']:7.3f} | {shift_info['status']}")
    

    print("\n" + "="*60)
    print("Cumulative Shift Details (relative to slice 0):")
    print("Slice | Cumulative X | Cumulative Y | Total Magnitude | Status")
    print("-"*60)
    
    for z in range(len(shifts)):
        dy, dx = shifts[z]
        magnitude = np.hypot(dx, dy)
        status = "❗ Bad" if z in bad_slices else "✅ Good"
        print(f"{z:5d} | {dx:14.2f} | {dy:14.2f} | {magnitude:15.2f} | {status}")