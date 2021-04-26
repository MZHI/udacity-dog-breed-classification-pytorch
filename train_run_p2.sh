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
#  --model-type Base_1_fix \
#  --prefix scratch_exp_35_wd_0.0005_aug2_w_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 36
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
#  --model-type vgg16 \
#  --prefix scratch_exp_36_wd_0.0005_aug2_w_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 37
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
#  --model-type vgg16 \
#  --prefix scratch_exp_37_wd_0.0005_aug2_wo_cj \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0

### experiment 38
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_38_wd_0.0005_augm1_fc_1_fix \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 1

### experiment 39
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_39_wd_0.0005_augm_cj_fc_2_fix \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 2

### experiment 40
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_40_wd_0.0005_augm_cj_fc_3_fix \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 3

### experiment 41
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_41_wd_0.0005_augm_wo_cj_fc_3_fix \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 3

### experiment 42
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 2 \
#  --num-classes 133 \
#  --lr 0.01 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type AlexNet \
#  --prefix pretrain_exp_42_wd_0.0005_augm_wo_cj_wo_hflip_fc_3_fix \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 3 \
#  --aug_h_flip 0

### experiment 43
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16 \
#  --prefix pretrain_exp_43_wd_0.0005_augm_wo_cj_wo_hflip_fc_1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 1 \
#  --aug_h_flip 0

### experiment 44
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16 \
#  --prefix pretrain_exp_44_wd_0.0005_augm_wo_cj_wo_hflip_fc_2 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 2 \
#  --aug_h_flip 0

### experiment 45
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16 \
#  --prefix pretrain_exp_45_wd_0.0005_augm_wo_cj_wo_hflip_fc_3 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 3 \
#  --aug_h_flip 0

### experiment 46
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16_bn \
#  --prefix pretrain_exp_46_wd_0.0005_augm_wo_cj_wo_hflip_fc_1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 1 \
#  --aug_h_flip 0

### experiment 47
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16_bn \
#  --prefix scratch_exp_47_wd_0.0005_augm_wo_cj_wo_hflip \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 0 \
#  --num-fc-train 1 \
#  --aug_h_flip 0

### experiment 48
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 1 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16_bn \
#  --prefix scratch_exp_48_wd_0.0005_augm_w_cj_wo_hflip \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 0 \
#  --num-fc-train 1 \
#  --aug_h_flip 0

### experiment 49
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16_bn \
#  --prefix scratch_exp_49_wd_0.0005_augm_w_cj_w_hflip \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 0 \
#  --num-fc-train 1 \
#  --aug_h_flip 1

### experiment 50
# python3 train.py --data-path data/dogImages \
#  --checkpoints-dir checkpoints \
#  --device 0 \
#  --log-path tensorboard_logs_p2 \
#  --batch-size 32 \
#  --num-epochs 500 \
#  --early-stopping 10 \
#  --num-workers 4 \
#  --num-classes 133 \
#  --lr 0.001 \
#  --optim SGD \
#  --weight-decay 0.0005 \
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16_bn \
#  --prefix pretrain_exp_50_wd_0.0005_augm_w_cj_w_hflip_fc_1 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 1 \
#  --num-fc-train 1 \
#  --aug_h_flip 1

### experiment 51
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
#  --color-jitter 0.4 0.4 0.4 0.2 \
#  --momentum 0.9 \
#  --dropout 0.5 \
#  --use-augm 1 \
#  --model-type vgg16_bn \
#  --prefix scratch_exp_51_wd_0.0005_augm_w_cj_w_hflip_custom_mean_std \
#  --mean 0.4864 0.4560 0.3918 \
#  --std 0.2602 0.2536 0.2562 \
#  --scheduler-patience 3 \
#  --scheduler-factor 0.5 \
#  --scheduler-cooldown 2 \
#  --save-last 1 \
#  --resume-train 0 \
#  --pretrained 0 \
#  --aug_h_flip 1

