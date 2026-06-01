import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np
from collections import defaultdict

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from src.models.graph.pointNet2.Graph_Model import PointNetSubNucleiNet
from src.datasets.graph.pointnet_dataset import PointNetNucleiDataset, pointnet_nuclei_collate_fn

def calculate_voi(pred_labels, gt_labels, eps=1e-10):
    
    if len(pred_labels) == 0:
        return 0.0, 0.0
    
    N = len(pred_labels)
    _, u_pred = np.unique(pred_labels, return_inverse=True)
    _, u_gt = np.unique(gt_labels, return_inverse=True)
    num_pred, num_gt = u_pred.max() + 1, u_gt.max() + 1
    

    count_matrix = np.zeros((num_pred, num_gt), dtype=np.float64)
    for p, g in zip(u_pred, u_gt):
        count_matrix[p, g] += 1
    

    p_pg = count_matrix / N
    p_p = p_pg.sum(axis=1)
    p_g = p_pg.sum(axis=0)
    

    p_p = p_p[p_p > 0]
    p_g = p_g[p_g > 0]
    p_pg = p_pg[p_pg > 0]
    

    h_p = -np.sum(p_p * np.log(p_p + eps))
    h_g = -np.sum(p_g * np.log(p_g + eps))
    h_pg = -np.sum(p_pg * np.log(p_pg + eps))
    
    voi_split = h_pg - h_g  # False splits
    voi_merge = h_pg - h_p  # False merges
    
    return voi_split, voi_merge

class UnionFind:
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, i):
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i == root_j:
            return

        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1
    
    def get_labels(self):
        return np.array([self.find(i) for i in range(len(self.parent))])

def get_segmentation_labels(edge_index_np, probs_np, num_nodes, threshold=0.5):
    
    uf = UnionFind(num_nodes)
    
    for i in range(len(probs_np)):
        if probs_np[i] > threshold:
            uf.union(edge_index_np[0, i], edge_index_np[1, i])
    
    return uf.get_labels()

def compute_contrastive_loss(node_features, gt_prototypes, node_gt_ids,
                             margin=2.0, temperature=0.1, normalize=False):
    device = node_features.device
    

    if normalize:
        node_features = F.normalize(node_features, p=2, dim=-1)
        gt_prototypes = F.normalize(gt_prototypes, p=2, dim=-1)
    

    _, inv_indices = torch.unique(node_gt_ids, return_inverse=True)
    

    target_centers = gt_prototypes[inv_indices]
    pull_loss = F.mse_loss(node_features, target_centers)
    

    num_prototypes = gt_prototypes.size(0)
    if num_prototypes > 1:
        dist_matrix = torch.cdist(gt_prototypes, gt_prototypes, p=2)
        mask = ~torch.eye(num_prototypes, device=device, dtype=torch.bool)
        diff_dist = dist_matrix[mask]
        push_loss = F.relu(margin - diff_dist).pow(2).mean()
    else:
        push_loss = torch.tensor(0.0, device=device)
    
    return pull_loss, push_loss

def validate(model, val_loader, device, args):
    
    model.eval()
    
    metrics = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            points = batch['points'].to(device)
            point_labels = batch['point_labels'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_gt = batch['edge_gt'].to(device)
            node_gt_ids = batch['node_gt_ids'].to(device)
            

            if edge_index is None or edge_index.shape[1] == 0:
                continue
            

            logits, node_features, gt_prototypes = model(
                points, point_labels, edge_index, node_gt_ids
            )
            

            cls_loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), edge_gt
            )
            metrics['cls_loss'].append(cls_loss.item())
            

            pull_loss, push_loss = compute_contrastive_loss(
                node_features, gt_prototypes, node_gt_ids, 
                margin=args.margin, normalize=args.normalize
            )
            feat_loss = pull_loss + push_loss
            metrics['feat_loss'].append(feat_loss.item())
            metrics['pull_loss'].append(pull_loss.item())
            metrics['push_loss'].append(push_loss.item())
            

            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()
            pred_labels = get_segmentation_labels(
                edge_index.cpu().numpy(), 
                probs, 
                len(node_gt_ids),
                threshold=args.seg_threshold
            )
            
            voi_split, voi_merge = calculate_voi(
                pred_labels, 
                node_gt_ids.cpu().numpy()
            )
            metrics['voi_split'].append(voi_split)
            metrics['voi_merge'].append(voi_merge)
            

            preds = (probs > 0.5).astype(np.float32)
            acc = (preds == edge_gt.cpu().numpy()).mean()
            metrics['edge_acc'].append(acc)
    

    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['voi_total'] = avg_metrics['voi_split'] + avg_metrics['voi_merge']
    
    return avg_metrics

def train(args):

    wandb.init(
        project="Nuclei-PointNet-Aggregation",
        name=args.exp_name,
        config=vars(args)
    )
    os.makedirs(args.save_dir, exist_ok=True)
    

    train_dataset = PointNetNucleiDataset(args.train_h5)
    val_dataset = PointNetNucleiDataset(args.val_h5)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pointnet_nuclei_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pointnet_nuclei_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    

    model = PointNetSubNucleiNet(
        point_feat_dim=args.point_feat_dim,
        gcn_hidden=args.gcn_hidden
    ).to(args.device)
    

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=0.5
        )
    else:
        scheduler = None
    

    if args.grad_clip > 0:
        clip_fn = lambda: torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip
        )
    else:
        clip_fn = lambda: None
    

    best_val_voi = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):

        model.train()
        train_metrics = defaultdict(list)
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]")
        
        for batch_idx, batch in enumerate(pbar):
            points = batch['points'].to(args.device)
            point_labels = batch['point_labels'].to(args.device)
            edge_index = batch['edge_index'].to(args.device)
            edge_gt = batch['edge_gt'].to(args.device)
            node_gt_ids = batch['node_gt_ids'].to(args.device)
            
            if edge_index is None or edge_index.shape[1] == 0:
                continue
            

            logits, node_features, gt_prototypes = model(
                points, point_labels, edge_index, node_gt_ids
            )
            

            cls_loss = F.binary_cross_entropy_with_logits(
                logits.view(-1), edge_gt
            )
            
            pull_loss, push_loss = compute_contrastive_loss(
                node_features, gt_prototypes, node_gt_ids,
                margin=args.margin, normalize=args.normalize
            )
            feat_loss = pull_loss + push_loss
            
            total_loss = cls_loss + args.alpha * feat_loss
            

            optimizer.zero_grad()
            total_loss.backward()
            clip_fn()
            optimizer.step()
            

            train_metrics['total_loss'].append(total_loss.item())
            train_metrics['cls_loss'].append(cls_loss.item())
            train_metrics['feat_loss'].append(feat_loss.item())
            train_metrics['pull_loss'].append(pull_loss.item())
            train_metrics['push_loss'].append(push_loss.item())
            

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.3f}",
                "Cls": f"{cls_loss.item():.3f}",
                "Pull": f"{pull_loss.item():.3f}",
                "Push": f"{push_loss.item():.3f}"
            })
        

        val_metrics = validate(model, val_loader, args.device, args)
        

        if scheduler is not None:
            scheduler.step()
        

        log_dict = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr']
        }
        

        for k, v in train_metrics.items():
            log_dict[f"train/{k}"] = np.mean(v)
        

        for k, v in val_metrics.items():
            log_dict[f"val/{k}"] = v
        
        wandb.log(log_dict)
        

        current_voi = val_metrics['voi_total']
        

        if current_voi < best_val_voi:
            best_val_voi = current_voi
            patience_counter = 0
            
            save_path = os.path.join(args.save_dir, "best_model_15000.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_voi': best_val_voi,
                'args': vars(args)
            }, save_path)
            
            print(f"\n🌟 New Best! VOI: {current_voi:.4f} "
                  f"(Split: {val_metrics['voi_split']:.4f}, "
                  f"Merge: {val_metrics['voi_merge']:.4f})")
        else:
            patience_counter += 1
        

        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_voi': current_voi,
                'args': vars(args)
            }, save_path)
        

        if args.early_stop > 0 and patience_counter >= args.early_stop:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    wandb.finish()
    print(f"\n✅ Training completed! Best VOI: {best_val_voi:.4f}")

def get_args():
    parser = argparse.ArgumentParser(description="PointNet++ Nuclei Segmentation Training")
    

    parser.add_argument('--train_h5', type=str, required=True, help="Path to training HDF5 dataset")
    parser.add_argument('--val_h5', type=str, required=True, help="Path to validation HDF5 dataset")
    parser.add_argument('--num_workers', type=int, default=8)
    

    parser.add_argument('--point_feat_dim', type=int, default=128)
    parser.add_argument('--gcn_hidden', type=int, default=128)
    

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=0.0,
                       help="")
    

    parser.add_argument('--alpha', type=float, default=1.0,
                       help="")
    parser.add_argument('--margin', type=float, default=5.0,
                       help="")
    parser.add_argument('--normalize', action='store_true',
                       help="")
    

    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'])
    parser.add_argument('--lr_decay_step', type=int, default=30,
                       help="")
    

    parser.add_argument('--seg_threshold', type=float, default=0.5,
                       help="")
    

    parser.add_argument('--save_dir', type=str, default="./checkpoints")
    parser.add_argument('--save_freq', type=int, default=10,
                       help="")
    parser.add_argument('--exp_name', type=str, default="PointNet2-Baseline")
    parser.add_argument('--early_stop', type=int, default=20,
                       help="")
    

    parser.add_argument('--device', type=str, default="cuda")
    
    return parser.parse_args()

if __name__ == "__main__":
    os.environ['PYTHONUNBUFFERED'] = '1'
    args = get_args()
    train(args)