#!/bin/bash

# 基础模型路径配置 (指向你的 UNet-BC best_model.pth)
MODEL_PATH="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/UNet/checkpoints_mouse/best_model_bc.pth"

# 结果保存根目录
RESULT_DIR="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/UNet/results/mouse"
mkdir -p $RESULT_DIR

# -------------------------------------------------------------------------
# 1. 评估 Merge Error 数据集中的 MS 样本 (Missing Segmentation)
# -------------------------------------------------------------------------
echo "Running UNet-BC evaluation on Merge Error (MS) samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error/ \
  --offsets_json /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error/slice_offsets.json \
  --seg_filter ms \
  --output_json $RESULT_DIR/batch_unet_results_ms.json \
  --model_path $MODEL_PATH \
  --iou_threshold 0.75 \
  --core_thresh 0.5 \
  --num_workers 2 \
  --device cuda:6

# -------------------------------------------------------------------------
# 2. 评估 Merge Error 数据集中的非 MS 样本 (真正的 Merge 错误)
# -------------------------------------------------------------------------
# echo "Running UNet-BC evaluation on Merge Error (Non-MS) samples..."
# python batch_correct.py \
#   --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error/ \
#   --offsets_json /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error/slice_offsets.json \
#   --seg_filter not_ms \
#   --output_json $RESULT_DIR/batch_unet_results_merge.json \
#   --model_path $MODEL_PATH \
#   --iou_threshold 0.75 \
#   --core_thresh 0.5 \
#   --num_workers 2 \
#   --device cuda:6

# -------------------------------------------------------------------------
# 3. 评估 Split Error 数据集
# -------------------------------------------------------------------------
echo "Running UNet-BC evaluation on Split Error samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/split_error \
  --offsets_json /nvme2/mingzhi/NucCorr/MICROS2Data/split_error/slice_offsets.json \
  --seg_filter all \
  --output_json $RESULT_DIR/batch_unet_results_split.json \
  --model_path $MODEL_PATH \
  --iou_threshold 0.75 \
  --core_thresh 0.5 \
  --num_workers 2 \
  --device cuda:6

echo "All UNet-BC evaluations completed!"