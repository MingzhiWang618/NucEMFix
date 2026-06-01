#!/bin/bash

# =========================================================================
# 1. 基础配置
# =========================================================================
# 指向你训练好的 Cellpose 模型（注意：Cellpose 通常指向模型文件名，不加 .pth）
MODEL_PATH="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/Cellpose/checkpoint_mouse/models/NucCorr_Cellpose_v3"

# 结果保存根目录
RESULT_DIR="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/Cellpose/results/mouse"
mkdir -p $RESULT_DIR

# 推理核心参数配置
ANISOTROPY=1.25    # Z轴与XY轴的比例
DIAMETER=10.0      # 细胞核平均直径
CELL_PROB=0.5      # 细胞概率阈值（越大越严格）
FLOW_THRES=0.5     # 梯度流阈值（越大越严格，减少合并）
DEVICE="cuda:3"    # 指定显卡

# =========================================================================
# 2. 评估任务流
# =========================================================================

# --- 任务 1: Merge Error 数据集中的 MS 样本 (Missing Segmentation) ---
echo "Running Cellpose evaluation on Merge Error (MS) samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error \
  --offsets_json slice_offsets.json \
  --seg_filter ms \
  --output_json $RESULT_DIR/batch_cellpose_results_ms.json \
  --model_path $MODEL_PATH \
  --anisotropy $ANISOTROPY \
  --diameter $DIAMETER \
  --cellprob_thresh $CELL_PROB \
  --flow_thresh $FLOW_THRES \
  --iou_threshold 0.75 \
  --device $DEVICE \
  --batch_size 8

# --- 任务 2: Merge Error 数据集中的非 MS 样本 (True Merge) ---
echo "Running Cellpose evaluation on Merge Error (Non-MS) samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error \
  --offsets_json slice_offsets.json \
  --seg_filter not_ms \
  --output_json $RESULT_DIR/batch_cellpose_results_merge.json \
  --model_path $MODEL_PATH \
  --anisotropy $ANISOTROPY \
  --diameter $DIAMETER \
  --cellprob_thresh $CELL_PROB \
  --flow_thresh $FLOW_THRES \
  --iou_threshold 0.75 \
  --device $DEVICE \
  --batch_size 8

# --- 任务 3: Split Error 数据集 ---
echo "Running Cellpose evaluation on Split Error samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/split_error \
  --offsets_json slice_offsets.json \
  --seg_filter all \
  --output_json $RESULT_DIR/batch_cellpose_results_split.json \
  --model_path $MODEL_PATH \
  --anisotropy $ANISOTROPY \
  --diameter $DIAMETER \
  --cellprob_thresh $CELL_PROB \
  --flow_thresh $FLOW_THRES \
  --iou_threshold 0.75 \
  --device $DEVICE \
  --batch_size 8

echo "===================================================="
echo "All Cellpose evaluations completed!"
echo "Results are saved in: $RESULT_DIR"