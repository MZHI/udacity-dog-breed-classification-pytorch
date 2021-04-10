#!/bin/bash

### experiment 26
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --mean 0.4864 0.4560 0.3918 \
#  --std 0.2602 0.2536 0.2562 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --model-type Base \
#  --prefix scratch_exp_26_wd_0.0005_augm1_wi_ones \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --weight-init-type ones

### experiment 27
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --mean 0.4864 0.4560 0.3918 \
#  --std 0.2602 0.2536 0.2562 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --model-type Base \
#  --prefix scratch_exp_27_wd_0.0005_augm1_wi_uni \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --weight-init-type uniform

### experiment 28
python3 train.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --log-path tensorboard_logs_p2 \
 --batch-size 32 \
 --num-epochs 500 \
 --early-stopping 10 \
 --num-workers 4 \
 --num-classes 133 \
 --lr 0.01 \
 --optim SGD \
 --weight-decay 0.0005 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --momentum 0.9 \
 --dropout 0.5 \
 --use-augm 1 \
 --color-jitter 0.4 0.4 0.4 0.2 \
 --model-type Base \
 --prefix scratch_exp_28_wd_0.0005_augm1_wi_gen \
 --scheduler-patience 3 \
 --scheduler-factor 0.5 \
 --scheduler-cooldown 2 \
 --save-last 1 \
 --resume-train 0 \
 --weight-init-type general

