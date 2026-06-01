#!/bin/bash

# 基礎模型路徑配置 (指向你的 nnUNet best_model.pth)
MODEL_PATH="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/nnUNet/checkpoints_fafb/best_model.pth"

# 結果保存根目錄
RESULT_DIR="/nvme2/mingzhi/NucCorr/TMI/OtherMethods/nnUNet/results/fafb"
mkdir -p $RESULT_DIR

# -------------------------------------------------------------------------
# 1. 評估 Merge Error 數據集中的 MS 樣本 (Missing Segmentation)
# -------------------------------------------------------------------------
echo "Running nnUNet-BC evaluation on Merge Error (MS) samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --offsets_json /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/slice_offsets.json \
  --seg_filter ms \
  --output_json $RESULT_DIR/batch_nnunet_results_ms.json \
  --model_path $MODEL_PATH \
  --iou_threshold 0.75 \
  --core_thresh 0.5 \
  --sw_batch_size 4 \
  --num_workers 2 \
  --device cuda:6

# -------------------------------------------------------------------------
# 2. 評估 Merge Error 數據集中的非 MS 樣本 (真正的 Merge 錯誤)
# -------------------------------------------------------------------------
echo "Running nnUNet-BC evaluation on Merge Error (Non-MS) samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14 \
  --offsets_json /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/slice_offsets.json \
  --seg_filter not_ms \
  --output_json $RESULT_DIR/batch_nnunet_results_merge.json \
  --model_path $MODEL_PATH \
  --iou_threshold 0.75 \
  --core_thresh 0.5 \
  --sw_batch_size 4 \
  --num_workers 2 \
  --device cuda:6

# -------------------------------------------------------------------------
# 3. 評估 Split Error 數據集
# -------------------------------------------------------------------------
echo "Running nnUNet-BC evaluation on Split Error samples..."
python batch_correct.py \
  --base_dir /nvme2/mingzhi/NucCorr/NucCorrData/split_error \
  --offsets_json /nvme2/mingzhi/NucCorr/NucCorrData/merge_error/merge_error_correct_6.14/slice_offsets.json \
  --seg_filter all \
  --output_json $RESULT_DIR/batch_nnunet_results_split.json \
  --model_path $MODEL_PATH \
  --iou_threshold 0.75 \
  --core_thresh 0.5 \
  --sw_batch_size 4 \
  --num_workers 2 \
  --device cuda:6

echo "All nnUNet-BC evaluations completed!"