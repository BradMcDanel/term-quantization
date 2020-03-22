#!/bin/bash

python evaluate_term_grouping.py /hdd1/datasets/ -a resnet18 -e --msgpack-loader --gpu 0 -b 64
python evaluate_term_grouping.py /hdd1/datasets/ -a mobilenet_v2 -e --msgpack-loader --gpu 0 -b 64
python evaluate_term_grouping.py /hdd1/datasets/ -a vgg16_bn -e --msgpack-loader --gpu 0 -b 64