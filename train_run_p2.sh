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
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
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
#  --prefix scratch_exp_28_wd_0.0005_augm1_wi_gen \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --weight-init-type general

### experiment 29
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
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
#  --prefix scratch_exp_29_wd_0.0005_augm2 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 

### experiment 30
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
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
#  --use-augm 0 \
#  --model-type Base_2 \
#  --prefix scratch_exp_30_wd_0.0005 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 

### experiment 31
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
#  --model-type Base_2 \
#  --prefix scratch_exp_31_wd_0.0005_aug_wo_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 

### experiment 32
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
#  --model-type Base_2 \
#  --prefix scratch_exp_32_wd_0.0005_aug_w_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 33
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
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
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix scratch_exp_33_wd_0.0005_augm2_w_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0
 
### experiment 34
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
#  --model-type Base_fix \
#  --prefix scratch_exp_34_wd_0.0005_aug2_w_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 35
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
 --model-type Base_1_fix \
 --prefix scratch_exp_35_wd_0.0005_aug2_w_cj \
 --scheduler-patience 3 \
 --scheduler-factor 0.5 \
 --scheduler-cooldown 2 \
 --save-last 1 \
 --resume-train 0
