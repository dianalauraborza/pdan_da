#!/usr/bin/env bash


python train_PDAN.py \
-dataset charades \
-mode rgb \
-model PDAN \
-train True \
-num_channel 512 \
-lr 0.0001 \
-comp_info charades_PDAN \
-APtype map \
-epoch 300 \
-batch_size 12 \
-num_summary_tokens 75 \
-rgb_root /data/stars/user/rdai/PhD_work/cvpr2020/Charades_v1/charades_feat_rgb
# -run_mode debug


