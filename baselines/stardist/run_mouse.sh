#!/bin/bash

# 基础模型路径配置
MODEL_DIR="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/models"
MODEL_NAME="stardist_mouse"

# 1. 评估 Merge Error 数据集中的 MS 样本
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error \
  --seg_filter ms \
  --output_json /nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/results/mouse/batch_stardist_results_ms.json \
  --offsets_json /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error/slice_offsets.json
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --iou_threshold 0.75 \
  --device cuda:3

# 2. 评估 Merge Error 数据集中的非 MS 样本 (通常是真正的 merge 错误)
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error \
  --seg_filter not_ms \
  --output_json /nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/results/mouse/batch_stardist_results_merge.json \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --iou_threshold 0.75 \
  --device cuda:3 \
  --offsets_json /nvme2/mingzhi/NucCorr/MICROS2Data/merge_error/slice_offsets.json

# 3. 评估 Split Error 数据集
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/MICROS2Data/split_error \
  --seg_filter all \
  --output_json /nvme2/mingzhi/NucCorr/TMI/OtherMethods/StarDist/results/mouse/batch_stardist_results_split.json \
  --model_dir $MODEL_DIR \
  --model_name $MODEL_NAME \
  --iou_threshold 0.75 \
  --device cuda:3 \
  --offsets_json /nvme2/mingzhi/NucCorr/MICROS2Data/split_error/slice_offsets.json