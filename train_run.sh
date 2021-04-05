#!/bin/bash

python3 train.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --log-path tensorboard_logs \
 --batch-size 32 \
 --num-epochs 2 \
 --early-stopping 10 \
 --num-workers 2 \
 --num-classes 133 \
 --lr 0.01 \
 --optim SGD \
 --momentum 0.9 \
 --dropout 0.5 \
 --model-type Base \
 --prefix scratch_exp_01 \
 --scheduler-patience 3 \
 --scheduler-factor 0.5 \
 --scheduler-cooldown 2 \
 --save-last 1 \
 --resume-train 0

