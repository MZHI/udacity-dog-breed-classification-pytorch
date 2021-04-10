#!/bin/bash

### experiment 1
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_01 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 2
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 64 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_02 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 3
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 128 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_03 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 4
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 16 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_04 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 5
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim Adam \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_05 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 6
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.00001 \
#  --optim Adam \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_06 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 7
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 256 \
#  --num-epochs 500 \
#  --early-stopping 20 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.00001 \
#  --optim Adam \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_07 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 8
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 256 \
#  --num-epochs 500 \
#  --early-stopping 20 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_08 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 9
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 256 \
#  --num-epochs 500 \
#  --early-stopping 20 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.1 \
#  --optim SGD \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_09 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 10
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --mean 0.4864 0.4560 0.3918 \
#  --std 0.2602 0.2536 0.2562 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --model-type Base \
#  --prefix scratch_exp_10 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 11
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
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
#  --model-type Base \
#  --prefix scratch_exp_11_wd_0.0005 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 12
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
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
#  --model-type Base \
#  --prefix scratch_exp_12_wd_0.0005_augm0 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 13
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
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
#  --model-type AlexNet \
#  --prefix scratch_exp_13_wd_0.0005_augm0 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 14
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
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
#  --model-type AlexNet \
#  --prefix scratch_exp_14_wd_0.0005 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 15
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
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
#  --prefix scratch_exp_15_wd_0.0005_augm1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 16
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
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
#  --prefix scratch_exp_16_wd_0.0005_augm1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 17
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_17_wd_0.0005_augm1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1

### experiment 18
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_18_wd_0.0005_augm1_fc_1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 1

### experiment 19
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_19_wd_0.0005_augm1_fc_2 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 2

### experiment 20
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_20_wd_0.0005_augm1_fc_3 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 3

### experiment 21
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
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
#  --model-type vgg16 \
#  --prefix scratch_exp_21_wd_0.0005_augm1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 0

### experiment 22
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --model-type vgg16 \
#  --prefix pretrain_exp_22_wd_0.0005_augm1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 1

### experiment 23
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --model-type vgg16 \
#  --prefix pretrain_exp_23_wd_0.0005_augm1_fc_2 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 2

### experiment 24
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --model-type vgg16 \
#  --prefix pretrain_exp_24_wd_0.0005_augm1_fc_3 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 3

### experiment 25
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs \
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
#  --model-type Base_1 \
#  --prefix scratch_exp_25_wd_0.0005_augm1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

