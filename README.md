# udacity-dog-breed-classification
Task from Udacity Deep Learning Nano Degree (see original task in `dog_app_orig.ipynb`). My way to implement and train CNN classifiers both from scratch and use PyTorch's implementations of popular CNNs for transfer learning.

# The purpose of this work
It is my own learning process of training convolution neural networks using PyTorch framework, from very beginning

# What interesting and useful can be found in this repo
* The code is well structured, scripts for training and evaluation are provided with wide range of input parameters
* A series of experiments are provided, well documented, see `experiments_part_1.ipynb`
* The tensorboard logs are provided, for each experiment you can see plots of train loss, validation loss, test loss and test accuracy. The logs are in `tensorboard_logs` directory
* You can view step-by-step on model's improvements and finetuning
* You can repeat all experiments

# How to use
1. The original task you can read in `dog_app_orig.ipynb`. You can also get links to dog breed dataset there
2. To see experiments description (from 1 to 25), run `experiments_part_1.ipynb`. Also, you can find conclusions and future work there
3. To run specific experiment, go to `train_run.sh` bash script, find experiment you are interested in, copy command and run in terminal
4. To evaluate model from specific experiment, go to `evaluate_run.sh` bash script, find experiment you are interested in, copy command and run in terminal

# Checkpoints
Checkpoints are not provided in this repo, but you can run all experiments from `train_run.sh` bash script

# Tensorboard logs
Tensorboard logs for all experiments are provided. To run tensorboard locally, run:
```sh
tensorboard -logdir tensorboard_logs/ 
```
To run tensorboard to get access from remote server, run (change [port] to your value):
```sh
tensorboard -logdir tensorboard_logs/ --port [port] --host 0.0.0.0
```

# Models description
Models are described in `experiments_part_1.ipynb`. Brief info: 
* two implementations of AlexNet from [original paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) (`Base` and `Base_1`), which are training from scratch 
* AlexNet from torchvision's implementation, which is training both from scratch and using transfer learning
* VGG16 from torchvision's implementation, which is training both from scratch and using transfer learning

# Train models from scratch
A lot of experiments are dedicated to train models from scratch. More detail information see in `experiments_part_1.ipynb`

# Transfer learning
Transfer learning are implemented for pretrained torchvision's models AlexNet and VGG16, with unfreezing from 1 to 3 last fully connected layers (unfreezing of convolution layers will be implemented in future)

# Feel free to use this work and have a good time this it :)
# If you have any ideas, or noticed any bugs, feel free to open issues, I'll be happy to improve this work!
