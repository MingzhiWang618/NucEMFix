import os
import numpy as np
import sys
from glob import glob
from tqdm import tqdm
from tifffile import imread
from stardist import fill_label_holes, random_label_cmap, calculate_extents
from stardist.models import Config3D, StarDist3D

def manual_normalize(x, pmin=1, pmax=99.8, axis=None, eps=1e-20):
    """
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return (x - mi) / (ma - mi + eps)

def main():

    base_dir = './data/supervised_data'
    X_paths = sorted(glob(os.path.join(base_dir, 'img/*.tiff')))
    Y_paths = sorted(glob(os.path.join(base_dir, 'seg/*.tiff')))

    assert all(os.path.basename(x.replace('_img', '')) == os.path.basename(y.replace('_seg', '')) for x, y in zip(X_paths, Y_paths))

    print(f"")
    X = [imread(x) for x in tqdm(X_paths, desc="Loading X")]
    Y = [imread(y) for y in tqdm(Y_paths, desc="Loading Y")]

    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0, 1, 2)

    print("")

    X = [manual_normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X, desc="Normalizing")]

    Y = [fill_label_holes(y) for y in tqdm(Y, desc="Filling holes")]

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]

    print(f'')

    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print(f"")

    conf = Config3D (
        rays             = 128,                # 3D Rays
        grid             = (2,2,2),
        train_patch_size = (96, 96, 96),
        anisotropy       = anisotropy,
        use_gpu          = True,
        n_channel_in     = n_channel,
        train_epochs     = 400,
        train_batch_size = 2,
    )

    model = StarDist3D(conf, name='stardist_nuclei', basedir='models')
    

    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val))

    print("")
    model.optimize_thresholds(X_val, Y_val)
    print("")

if __name__ == '__main__':
    main()