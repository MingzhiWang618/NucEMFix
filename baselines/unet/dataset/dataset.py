import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tifffile import imread
from skimage.segmentation import find_boundaries
import random

def normalize_percentile(x, pmin=1, pmax=99.8, axis=None, eps=1e-20):
    """
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return (x - mi) / (ma - mi + eps)

def convert_to_bc_mask(instance_mask):
    """
    """

    bc_mask = np.zeros_like(instance_mask, dtype=np.int64)
    

    bc_mask[instance_mask > 0] = 1
    

    boundaries = find_boundaries(instance_mask, mode='inner', background=0)
    

    bc_mask[boundaries] = 2
    
    return bc_mask

class BCDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, patch_size=(128, 128, 128)):
        """
        Args:
        """
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        

        img = imread(img_path).astype(np.float32)
        mask = imread(mask_path)

        img = normalize_percentile(img)

        bc_mask = convert_to_bc_mask(mask)

        if self.transform:
            img, bc_mask = self._random_augment(img, bc_mask)

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()

        mask_tensor = torch.from_numpy(bc_mask).long()

        return img_tensor, mask_tensor

    def _random_augment(self, img, mask):
        """
        """

        if random.random() > 0.5:
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
        

        if random.random() > 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)
            

        if random.random() > 0.5:
            img = np.flip(img, axis=2)
            mask = np.flip(mask, axis=2)

        k = random.randint(0, 3)
        img = np.rot90(img, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(1, 2))

        return img.copy(), mask.copy()

if __name__ == "__main__":
    from glob import glob
    

    base_dir = './data/supervised_data'
    X_paths = sorted(glob(os.path.join(base_dir, 'img/*.tiff')))
    Y_paths = sorted(glob(os.path.join(base_dir, 'seg/*.tiff')))

    if len(X_paths) > 0:
        dataset = BCDataset(X_paths, Y_paths, transform=True)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print(f"Dataset length: {len(dataset)}")
        

        img, mask = next(iter(dataloader))
        print(f"Image Batch Shape: {img.shape}")
        print(f"Mask Batch Shape: {mask.shape}")
        print(f"Unique labels in mask: {torch.unique(mask)}")
    else:
        print("")