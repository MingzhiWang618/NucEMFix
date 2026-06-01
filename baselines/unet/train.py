import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from glob import glob
from tqdm import tqdm

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from baselines.unet.model.model import UNet3D_BC
from baselines.unet.dataset.dataset import BCDataset

def train_model():

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    base_dir = './data/supervised_data'
    model_save_path = './checkpoints_fafb'
    os.makedirs(model_save_path, exist_ok=True)

    X_paths = sorted(glob(os.path.join(base_dir, 'img/*.tiff')))
    Y_paths = sorted(glob(os.path.join(base_dir, 'seg/*.tiff')))
    
    if len(X_paths) == 0:
        print(f"")
        return

    full_dataset = BCDataset(X_paths, Y_paths, transform=True)
    

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet3D_BC(in_channels=1, n_classes=3).to(device)
    

    weights = torch.tensor([1.0, 1.0, 10.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 400
    best_val_loss = float('inf')

    print(f"")
    
    for epoch in range(num_epochs):

        model.train()
        running_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            

            optimizer.zero_grad()
            

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            

            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model_bc.pth'))
            print("")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    train_model()