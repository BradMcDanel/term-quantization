from __future__ import print_function

import argparse
import os
from pprint import pprint

import numpy as np
import torch
import math
import torch.backends.cudnn as cudnn
cudnn.benchmark = True 
import torch.optim as optim
import torch.nn as nn

import datasets
import util
from mobilenetv2 import MobileNetV2
from vgg import VGG

loss = nn.CrossEntropyLoss()

def train_model(model, model_path, train_loader, test_loader, init_lr, epochs):
    # tracking stats
    if not hasattr(model, 'stats'):
        model.stats = {'train_loss': [], 'test_acc': [], 'test_loss': [], 'lr': []}

    # optimizer
    ps = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(ps, lr=init_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    print("Optimizer:")
    print(optimizer)

    # pruning stage
    for epoch in range(1, epochs + 1):
        print('[Epoch {}]'.format(epoch))
        util.adjust_learning_rate(optimizer, epoch - 1, init_lr)
        for g in optimizer.param_groups:     
            lr = g['lr']                    
            break        

        train_loss = util.train(model, train_loader, optimizer, epoch, loss)
        test_loss, test_acc = util.test(model, test_loader, epoch, loss)

        print('LR        :: {}'.format(lr))
        print('Train Loss:: {}'.format(train_loss))
        print('Test  Loss:: {}'.format(test_loss))
        print('Test  Acc.:: {}'.format(test_acc))
        model.stats['train_loss'].append(train_loss)
        model.stats['test_loss'].append(test_loss)
        model.stats['test_acc'].append(test_acc)
        model.stats['lr'].append(lr)
        model.optimizer = optimizer.state_dict()

        model.cpu()
        torch.save(model, model_path)
        model.cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Training Script')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--aug', default='+', help='data augmentation level (`-`, `+`)')
    parser.add_argument('--load-path', default=None,
                        help='path to load model - trains new model if None')
    parser.add_argument('--in-memory', action='store_true',
                        help='ImageNet Dataloader setting (store in memory)')
    parser.add_argument('--input-size', type=int, help='spatial width/height of input')
    parser.add_argument('--n-class', type=int, help='number of classes')
    parser.add_argument('--save-path', required=True, help='path to save model')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print('Arguments:')
    pprint(args.__dict__, width=1)

    #set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load dataset
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size,
                                args.cuda, args.aug, in_memory=args.in_memory,
                                input_size=args.input_size)
    train_dataset, train_loader, test_dataset, test_loader = data

    # load or create model
    if args.load_path == None:
        # model = MobileNetV2(num_classes=args.n_class)
        model = VGG('VGG11', act='relu', group_size=8)
    else:
        model = torch.load(args.load_path)

    if args.cuda:
        model = model.cuda()

    print(model)

    train_model(model, args.save_path, train_loader, test_loader, args.lr, args.epochs)