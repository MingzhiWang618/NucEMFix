#!/bin/bash

# 基础模型路径配置
MODEL_DIR="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/models"
MODEL_NAME="stardist_fafb"

# 1. 评估 Merge Error 数据集中的 MS 样本
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter ms \
  --offsets_json /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/slice_offsets.json \
  --output_json /nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/results/fafb/batch_stardist_results_ms.json \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --iou_threshold 0.75 \
  --device cuda:2

# 2. 评估 Merge Error 数据集中的非 MS 样本 (通常是真正的 merge 错误)
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --seg_filter not_ms \
  --offsets_json /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/slice_offsets.json \
  --output_json /nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/results/fafb/batch_stardist_results_merge.json \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --iou_threshold 0.75 \
  --device cuda:2

# 3. 评估 Split Error 数据集
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --seg_filter all \
  --offsets_json /nvme2/mingzhi/NucCorr/NucCorrData/split_error/slice_offsets.json \
  --output_json /nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/results/fafb/batch_stardist_results_split.json \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --iou_threshold 0.75 \
  --device cuda:2