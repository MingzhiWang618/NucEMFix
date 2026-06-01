import os
import torch
import numpy as np
from glob import glob
from tifffile import imread as tiff_imread
from tqdm import tqdm
from torch.utils.data import random_split

from monai.transforms import (
    Compose, 
    NormalizeIntensityd, 
    RandFlipd, 
    RandSpatialCropd,
    ToTensord, 
    MapTransform
)
from monai.networks.nets import DynUNet
from monai.data import Dataset, DataLoader
from skimage.segmentation import find_boundaries

class LoadTiffd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = tiff_imread(d[key]) 
            d[key] = img[np.newaxis, ...].astype(np.float32 if key == "image" else np.int32)
        return d

class ConvertToBCd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]
            is_tensor = torch.is_tensor(label)
            work_label = label[0].cpu().numpy() if is_tensor else label[0]
            
            boundaries = find_boundaries(work_label, mode='inner', background=0)
            bc_label = np.zeros_like(work_label, dtype=np.uint8)
            bc_label[work_label > 0] = 1   # Core
            bc_label[boundaries] = 2       # Boundary
            
            new_label = bc_label[None, ...]
            d[key] = torch.from_numpy(new_label) if is_tensor else new_label
        return d

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda")

    img_dir = "./data/img"
    seg_dir = "./data/seg"
    model_dir = "./checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    img_files = sorted(glob(os.path.join(img_dir, "*.tiff")))
    seg_files = sorted(glob(os.path.join(seg_dir, "*.tiff")))
    data_dicts = [{"image": i, "label": s} for i, s in zip(img_files, seg_files)]

    train_size = int(0.85 * len(data_dicts))
    val_size = len(data_dicts) - train_size
    train_files, val_files = random_split(data_dicts, [train_size, val_size])

    train_transforms = Compose([
        LoadTiffd(keys=["image", "label"]),
        ConvertToBCd(keys=["label"]),
        NormalizeIntensityd(keys=["image"]),
        RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_center=True, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadTiffd(keys=["image", "label"]),
        ConvertToBCd(keys=["label"]),
        NormalizeIntensityd(keys=["image"]),
        RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_center=True, random_size=False),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    model = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=3,
        kernel_size=[[3,3,3]]*5, strides=[[1,1,1]] + [[2,2,2]]*4,
        upsample_kernel_size=[[2,2,2]]*4, norm_name="instance", deep_supervision=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 10.0]).to(device))

    start_epoch = 0
    checkpoint_path = os.path.join(model_dir, "latest_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"")

    num_epochs = 100
    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_data in pbar:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            labels = labels.squeeze(1).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                labels = labels.squeeze(1).long()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(f"")

if __name__ == "__main__":
    main()