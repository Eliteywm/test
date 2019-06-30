#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

./train.py --arch="mobilenet_1" \
    --start-epoch=1 \
    --loss=wpdc \
    --snapshot="/home/ai/3d/train.configs/snapshot/phase1_wpdc" \
    --param-fp-train='/home/ai/3d/train.configs/param_all_norm.pkl' \
    --param-fp-val='/home/ai/3d/train.configs/param_all_norm_val.pkl' \
    --warmup=5 \
    --opt-style=resample \
    --resample-num=132 \
    --batch-size=80 \
    --base-lr=0.02 \
    --epochs=50 \
    --milestones=30,40 \
    --print-freq=50 \
    --devices-id=0 \
    --workers=8 \
    --filelists-train="/home/ai/3d/train.configs/train_aug_120x120.list.train" \
    --filelists-val="/home/ai/3d/train.configs/train_aug_120x120.list.val" \
    --root="/home/ai/3d/train.configs/train_aug_120x120" \
    --log-file="${LOG_FILE}"
