from copy import deepcopy
import argparse
import json

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import thop

from train_mlp import MNISTMLP, test
from tr_layer import TRLinearLayer, set_tr_tracking
from profile_model import get_model_ops

def replace_linear_layers(model, tr_params, data_bits, data_terms):
    curr_layer = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            weight_bits, group_size, weight_terms = tr_params[curr_layer]
            layer = TRLinearLayer(layer, data_bits, data_terms, weight_bits,
                                  group_size, weight_terms)

            module._modules[module_keys[-1]] = layer
            curr_layer += 1

    return model

def static_linear_layer_settings(model, weight_bits, group_size, num_terms):
    curr_layer = 0
    stats = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            stats.append((weight_bits, group_size, num_terms))
            curr_layer += 1

    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--wb', nargs='+', type=int, help='weight bits')
    parser.add_argument('--wt', nargs='+', type=int, help='weight terms')
    parser.add_argument('--db', nargs='+', type=int, help='data bits')
    parser.add_argument('--dt', nargs='+', type=int, help='data terms')
    parser.add_argument('--gs', nargs='+', type=int, help='group sizes')
    parser.add_argument('--out-file', help='Output file')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = MNISTMLP()
    state_dict = torch.load('data/mnist_mlp.pt')
    model.load_state_dict(state_dict)

    settings = zip(args.wb, args.wt, args.db, args.dt, args.gs)

    results = {'accs': [], 'tmacs': [], 'param_bits': []}
    for wb, wt, db, dt, gs in settings:
        # Convert model
        qmodel = deepcopy(model)
        qmodel.to(device)
        tr_params = static_linear_layer_settings(qmodel, wb, gs, wt)
        qmodel = replace_linear_layers(qmodel, tr_params, db, dt)

        # Profile
        acc = test(args, qmodel, device, test_loader, pct=0.05)
        set_tr_tracking(qmodel, False)

        # Get results
        acc = test(args, qmodel, device, test_loader)
        acc = 100.0 * acc
        tmacs, param_bits = get_model_ops(qmodel, input_shape=(1, 1, 28, 28))
        results['accs'].append(acc)
        results['tmacs'].append(tmacs)
        results['param_bits'].append(param_bits)
        print(wb, wt, db, dt, gs, acc, tmacs, param_bits)

    with open(args.out_file, 'w') as fp:
        json.dump(results, fp)
