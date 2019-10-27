# Superposition of many models into one

This repository contains the source code for the paper: [Superposition of many models into one](https://arxiv.org/abs/1902.05522).

### Requirements before running
This code requires:
* python 3.\*
* numpy
* pytorch
* torchvision
* tensorboardx
* tensorboard
* matplotlib
* scikit-image

### Usage

#### 1. Run the code
Experiments for the paper are executed in 'main_superposition.py' 

```bash
$ python main_superposition.py
```

#### 2. View the results
Results are displayed on in the 'runs' folder and are readable via

```bash
$ tensorboard --logdir=./runs
```
