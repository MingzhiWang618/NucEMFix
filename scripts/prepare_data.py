#!/usr/bin/env python
import os
import json
import tifffile as tiff
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_dataset(input_dir, output_dir, train_ratio=0.8):
    """Prepare dataset for training and testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory structure
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, 'img'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'seg'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'correct'), exist_ok=True)
    
    # Get all sample IDs
    img_dir = os.path.join(input_dir, 'img')
    seg_dir = os.path.join(input_dir, 'seg')
    correct_dir = os.path.join(input_dir, 'correct')
    
    sample_ids = []
    for f in os.listdir(img_dir):
        if f.endswith('.tiff'):
            match = re.search(r'\d+', f)
            if match:
                sample_ids.append(match.group())
    
    # Split into train/val/test
    train_ids, test_ids = train_test_split(sample_ids, train_size=train_ratio, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, train_size=0.5, random_state=42)
    
    # Copy files
    def copy_files(ids, dest_dir):
        for fid in ids:
            # Find corresponding files
            img_files = [f for f in os.listdir(img_dir) if fid in f and f.endswith('.tiff')]
            seg_files = [f for f in os.listdir(seg_dir) if fid in f and f.endswith('.tiff')]
            correct_files = [f for f in os.listdir(correct_dir) if fid in f and f.endswith('.tiff')]
            
            if img_files and seg_files and correct_files:
                os.symlink(os.path.join(img_dir, img_files[0]), 
                          os.path.join(dest_dir, 'img', img_files[0]))
                os.symlink(os.path.join(seg_dir, seg_files[0]), 
                          os.path.join(dest_dir, 'seg', seg_files[0]))
                os.symlink(os.path.join(correct_dir, correct_files[0]), 
                          os.path.join(dest_dir, 'correct', correct_files[0]))
    
    copy_files(train_ids, train_dir)
    copy_files(val_ids, val_dir)
    copy_files(test_ids, test_dir)
    
    # Save split information
    split_info = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    with open(os.path.join(output_dir, 'dataset_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Dataset prepared successfully!")
    print(f"Train: {len(train_ids)} samples")
    print(f"Val: {len(val_ids)} samples")
    print(f"Test: {len(test_ids)} samples")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare dataset for NucEMFix')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for prepared dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training samples')
    args = parser.parse_args()
    
    prepare_dataset(args.input_dir, args.output_dir, args.train_ratio)