#!/bin/bash

python evaluate_cnn.py /hdd1/datasets/ -a resnet18 --gpu 0 -b 64
python evaluate_cnn.py /hdd1/datasets/ -a vgg16_bn --gpu 0 -b 64
python evaluate_cnn.py /hdd1/datasets/ -a mobilenet_v2 --gpu 0 -b 64
python evaluate_cnn.py /hdd1/datasets/ -a efficientnet_b0 --gpu 0 -b 64