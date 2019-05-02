import argparse

import torch
import torch.nn as nn


import datasets
from group_relu import GroupReLU
from torchvision.utils import make_grid, save_image

class Tracker(nn.Module):
    def __init__(self):
        super(Tracker, self).__init__()
        self.x = None
    
    def forward(self, x):
        self.x = x
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Training Script')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--aug', default='+', help='data augmentation level (`-`, `+`)')
    parser.add_argument('--load-path', default=None, required=True,
                        help='path to load model - trains new model if None')
    parser.add_argument('--in-memory', action='store_true',
                        help='ImageNet Dataloader setting (store in memory)')
    parser.add_argument('--input-size', type=int, help='spatial width/height of input')
    parser.add_argument('--n-class', type=int, help='number of classes')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    #set random seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # load dataset
    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size,
                                args.cuda, args.aug, in_memory=args.in_memory,
                                input_size=args.input_size)
    train_dataset, train_loader, test_dataset, test_loader = data
    model = torch.load(args.load_path)

    if args.cuda:
        model = model.cuda()

    for i, layer in enumerate(model.features.children()): 
        if type(layer) == GroupReLU or type(layer) == nn.ReLU:
            model.features[i] = nn.Sequential(model.features[i], Tracker())

    for i, (data, target) in enumerate(test_loader):
        data = data.cuda()
        model(data)
        break

if type(model.features[2][0]) == nn.ReLU:
    gs = 1
else:
    gs = model.features[2][0].group_size

t = model.features[2][1].x
B, C, W, H = t.shape
save_image(make_grid(t[1].unsqueeze(1).cpu()), 'figures/test.png')

t = t.permute(2, 3, 0, 1).contiguous().view(-1, gs)
tt = t.max(1)[0]
tt = tt.view(W, H, B, C // gs)
tt = tt.permute(2, 3, 0, 1).contiguous()
save_image(make_grid(tt[1].unsqueeze(1).cpu(), nrow=1), 'figures/test-sum.png')