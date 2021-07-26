#!/usr/bin/env bash

export PATH=/Your/ENV/bin:$PATH

python train.py \
-dataset charades \
-mode rgb \
-model PDAN \
-train True \
-num_channel 512 \
-lr 0.001 \
-kernelsize 9 \
-comp_info charades_PDAN \
-APtype map \
-epoch 100 \
-batch_size 16 # -run_mode debug
