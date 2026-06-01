import numpy as np
import cc3d
from tqdm import tqdm
import numpy as np
from skimage.segmentation import watershed
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def watershed_Oversegmentation_distance(
    seg,
    min_size=100,
    min_distance=3
):
    """
    Instance-wise over-segmentation using distance-transform-driven watershed.

    For each instance in seg:
        1. Compute distance transform inside the instance mask
        2. Extract local maxima as markers
        3. Apply marker-controlled watershed on -distance
        4. Merge small regions using adaptive size threshold

    Parameters
    ----------
    image : ndarray
        Unused in watershed stage, kept for interface compatibility
    seg : ndarray (int)
        Instance segmentation mask (3D)
    min_size : int
        Minimum size of a valid instance
    min_distance : int
        Minimum distance between distance-transform peaks

    Returns
    -------
    out : ndarray (int)
        Over-segmented instance mask
    """

    unique_ids = np.unique(seg)
    unique_ids = unique_ids[unique_ids > 0]

    out = np.zeros_like(seg, dtype=np.int32)
    label_offset = 1

    for id_val in tqdm(unique_ids, desc="Processing instances"):
        id_mask = (seg == id_val)

        # Safety: split disconnected parts inside one instance ID
        cc = cc3d.connected_components(id_mask, connectivity=6)

        for cc_label in range(1, cc.max() + 1):
            region_mask = (cc == cc_label)
            region_size = np.count_nonzero(region_mask)
            if region_size < min_size:
                continue

            # --- Crop ROI ---
            coords = np.argwhere(region_mask)
            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0) + 1

            seg_crop = region_mask[zmin:zmax, ymin:ymax, xmin:xmax]

            # --- Distance transform ---
            distance_map = distance_transform_edt(seg_crop)

            # --- Marker generation (distance peaks) ---
            local_max = peak_local_max(
                distance_map,
                min_distance=min_distance,
                labels=seg_crop,
                exclude_border=False,
                footprint=np.ones((8, 8, 8))
            )

            # Fallback: ensure at least one marker
            if len(local_max) == 0:
                center = np.array(seg_crop.shape) // 2
                local_max = np.array([center])

            markers = np.zeros_like(seg_crop, dtype=np.int32)
            for i, (z, y, x) in enumerate(local_max, start=1):
                markers[z, y, x] = i

            # --- Watershed on negative distance ---
            labels_ws = watershed(
                -distance_map,
                markers=markers,
                mask=seg_crop
            )

            # --- Adaptive region merging ---
            region_sizes = [
                np.count_nonzero(labels_ws == sub_id)
                for sub_id in range(1, labels_ws.max() + 1)
            ]

            if not region_sizes:
                continue

            median_size = np.median(region_sizes)
            min_region_size = max(min_size, int(0.3 * median_size))

            final_labels = np.zeros_like(labels_ws, dtype=np.int32)
            curr_label = 1

            # Keep large regions
            for sub_id in range(1, labels_ws.max() + 1):
                mask_sub = (labels_ws == sub_id)
                sub_size = np.count_nonzero(mask_sub)

                if sub_size < min_region_size:
                    continue

                final_labels[mask_sub] = curr_label
                curr_label += 1

            # Merge small regions into neighbors
            unprocessed = (final_labels == 0) & (labels_ws > 0)
            if np.any(unprocessed):
                struct = np.ones((3, 3, 3))
                dilated = binary_dilation(unprocessed, structure=struct)
                neighbors = final_labels * dilated

                for z, y, x in np.argwhere(unprocessed):
                    local = neighbors[
                        max(0, z - 1):min(z + 2, neighbors.shape[0]),
                        max(0, y - 1):min(y + 2, neighbors.shape[1]),
                        max(0, x - 1):min(x + 2, neighbors.shape[2])
                    ]
                    local = local[local > 0]

                    if local.size > 0:
                        values, counts = np.unique(local, return_counts=True)
                        final_labels[z, y, x] = values[np.argmax(counts)]
                    else:
                        final_labels[z, y, x] = labels_ws[z, y, x]

            # --- Write back to output ---
            for sub_id in range(1, final_labels.max() + 1):
                mask_sub = (final_labels == sub_id)
                sub_size = np.count_nonzero(mask_sub)

                if sub_size >= min_size:
                    out[zmin:zmax, ymin:ymax, xmin:xmax][mask_sub] = label_offset
                    label_offset += 1

    return out

