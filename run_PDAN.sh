#!/usr/bin/env bash

export PATH=/Your/ENV/bin:$PATH

python train_PDAN.py \
-dataset charades \
-mode rgb \
-model PDAN \
-train True \
-num_channel 512 \
-lr 0.0001 \
-comp_info charades_PDAN \
-APtype map \
-epoch 100 \
-batch_size 32 \
-lr 0.001 \
-num_summary_tokens 32 \
-rgb_root /data/stars/user/rdai/PhD_work/cvpr2020/Charades_v1/charades_feat_rgb
# -run_mode debug

