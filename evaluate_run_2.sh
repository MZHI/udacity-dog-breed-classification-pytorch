#!/bin/bash

### experiment 26
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_26

### experiment 27
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_27
 
### experiment 28
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_28

### experiment 29
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base \
 --exp exp_29

# experiment 30
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base_2 \
 --exp exp_30

# experiment 31
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base_2 \
 --exp exp_31

# experiment 32
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base_2 \
 --exp exp_32

# experiment 33
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_33

# experiment 34
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --num-classes 133 \
 --model-type Base_fix \
 --exp exp_34

# experiment 35
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 2 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type Base_1_fix \
 --exp exp_35

# experiment 36
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type vgg16 \
 --exp exp_36

# experiment 37
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type vgg16 \
 --exp exp_37

# experiment 38
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_38

# experiment 39
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_39

# experiment 40
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_40

# experiment 41
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_41

# experiment 42
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type AlexNet \
 --exp exp_42

# experiment 43
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16 \
 --exp exp_43

# experiment 44
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16 \
 --exp exp_44

# experiment 45
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16 \
 --exp exp_45 \

# experiment 46
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16_bn \
 --exp exp_46

# experiment 47
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16_bn \
 --exp exp_47

# experiment 48
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16_bn \
 --exp exp_48

# experiment 49
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 1 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16_bn \
 --exp exp_49

### experiment 50
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --model-type vgg16_bn \
 --exp exp_50
 
### experiment 51
python3 test.py --data-path data/dogImages \
 --checkpoints-dir checkpoints \
 --device 0 \
 --batch-size 32 \
 --num-workers 4 \
 --num-classes 133 \
 --mean 0.4864 0.4560 0.3918 \
 --std 0.2602 0.2536 0.2562 \
 --model-type vgg16_bn \
 --exp exp_51

