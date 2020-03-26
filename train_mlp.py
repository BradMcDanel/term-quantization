from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import sklearn.metrics as metrics

import numpy as np
import pandas as pd
from scipy import io

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

class ToxDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def load_tox21(args):
    y_tr = pd.read_csv(args.val_dir + 'tox21/tox21_labels_train.csv.gz',
                       index_col=0, compression="gzip")
    y_te = pd.read_csv(args.val_dir + 'tox21/tox21_labels_test.csv.gz',
                       index_col=0, compression="gzip")
    x_tr_dense = pd.read_csv(args.val_dir+ 'tox21/tox21_dense_train.csv.gz',
                             index_col=0, compression="gzip").values
    x_te_dense = pd.read_csv(args.val_dir + 'tox21/tox21_dense_test.csv.gz',
                             index_col=0, compression="gzip").values
    x_tr_sparse = io.mmread(args.val_dir + 'tox21/tox21_sparse_train.mtx.gz').tocsc()
    x_te_sparse = io.mmread(args.val_dir + 'tox21/tox21_sparse_test.mtx.gz').tocsc()

    sparse_col_idx = ((x_tr_sparse > 0).mean(0) > 0.05).A.ravel()
    x_tr = np.hstack([x_tr_dense, x_tr_sparse[:, sparse_col_idx].A])
    x_te = np.hstack([x_te_dense, x_te_sparse[:, sparse_col_idx].A])

    return tuple((x_tr, y_tr, x_te, y_te)), y_tr.columns

def build_target(args, target, data):
    x_tr, y_tr, x_te, y_te = data
    mask = np.isfinite(y_tr[target]).values
    x_tr = torch.Tensor(x_tr[mask])
    y_tr = torch.Tensor(y_tr[target][mask]).long()
    mask = np.isfinite(y_te[target]).values
    x_te = torch.Tensor(x_te[mask])
    y_te = torch.Tensor(y_te[target][mask]).long()

    print(x_tr.shape)
    means = x_tr.mean(0)
    stds = x_tr.std(0)

    x_tr = (x_tr - means) / (stds+1e-6)
    x_te = (x_te - means) / (stds+1e-6)

    train = ToxDataset(x_tr, y_tr)
    test = ToxDataset(x_te, y_te)

    train_loader = torch.utils.data.DataLoader(train, args.batch_size)
                                              
    test_loader = torch.utils.data.DataLoader(test, args.batch_size)

    return train_loader, test_loader

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(1644, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x.flatten(1))
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    auc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            auc += metrics.roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    return auc / len(test_loader.dataset)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('val_dir', help='path to validation data folder')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    data, targets = load_tox21(args)
    device = torch.device("cuda" if use_cuda else "cpu")

    for target in targets:
        train_loader, test_loader = build_target(args, target, data)
        model = MLP().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            auc = test(args, model, device, test_loader)
            print(auc)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "data/tox_{}.pt".format(target))
        assert False