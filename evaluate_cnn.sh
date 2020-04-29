#!/bin/bash

python evaluate_cnn.py /hdd1/datasets/ -a resnet18 -e --gpu 0 -b 64
python evaluate_cnn.py /hdd1/datasets/ -a vgg16_bn -e --gpu 0 -b 64
python evaluate_cnn.py /hdd1/datasets/ -a mobilenet_v2 -e --gpu 0 -b 64
python evaluate_cnn.py /hdd1/datasets/ -a efficientnet_b0 -e --gpu 0 -b 64