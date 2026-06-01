import os
import numpy as np
from cellpose import models, io, train

# ==========================================

# ==========================================
TRAIN_DIR = './data/cellpose/train'
VAL_DIR = './data/cellpose/val'
OUTPUT_DIR = './checkpoints/cellpose'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_training():
    io.logger_setup()

    model = models.CellposeModel(gpu=True, model_type='nuclei')

    print("")
    train_data, train_labels, _, test_data, test_labels, _ = io.load_train_test_data(
        TRAIN_DIR, 
        test_dir=VAL_DIR, 
        mask_filter='_masks'
    )
    
    print(f"")

    print("")
    

    device = model.device

    model_path = train.train_seg(
        model.net,
        train_data=train_data, 
        train_labels=train_labels, 
        test_data=test_data, 
        test_labels=test_labels,
        channels=[0, 0],
        batch_size=64, 
        n_epochs=500, 
        learning_rate=0.001, 
        weight_decay=0.0001, 
        save_path=OUTPUT_DIR, 
        model_name='NucCorr_Cellpose_v3'
    )

    print(f"")

if __name__ == '__main__':
    run_training()