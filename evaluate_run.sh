#!/bin/bash

### experiment 1
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_01

### experiment 2
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_02
 
### experiment 3
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_03

### experiment 4
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 16 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_04

# experiment 5
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_05

# experiment 6
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --num-classes 133 \
 --lr 0.00001 \
 --model-type Base \
 --exp exp_06

# experiment 7
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_07

# experiment 8
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_08

# experiment 9
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_09

# experiment 10
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type Base \
 --exp exp_10

# experiment 11
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type Base \
 --exp exp_11

# experiment 12
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type Base \
 --exp exp_12

# experiment 13
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type AlexNet \
 --exp exp_13

# experiment 14
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type AlexNet \
 --exp exp_14

# experiment 15
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type AlexNet \
 --exp exp_15

# experiment 16
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type Base \
 --exp exp_16

# experiment 17
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_17

# experiment 18
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_18

# experiment 19
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_19

# experiment 20
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_20 \

# experiment 21
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type vgg16 \
 --exp exp_21

# experiment 22
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16 \
 --exp exp_22

# experiment 23
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16 \
 --exp exp_23

# experiment 24
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16 \
 --exp exp_24

### experiment 25
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type Base_1 \
 --exp exp_25
 
