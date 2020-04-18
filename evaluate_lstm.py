import argparse
import math
from copy import deepcopy
import json

import torch
import torch.nn as nn

import lstm_models.model as model
import lstm_models.data as data
from tr_layer import TRLSTMLayer, TRLinearLayer, set_tr_tracking
import profile_model

import sys
sys.path.insert(0, './lstm_models')

def replace_lstm_layers(model, tr_params, data_bits, data_terms):
    curr_layer = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.LSTM)):
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            weight_bits, group_size, weight_terms = tr_params[curr_layer]
            if isinstance(layer, nn.LSTM):
                layer = TRLSTMLayer(layer, data_bits, data_terms, weight_bits,
                                    group_size, weight_terms)
            elif isinstance(layer, nn.Linear):
                layer = TRLinearLayer(layer, data_bits, data_terms, weight_bits,
                                      group_size, weight_terms)

            module._modules[module_keys[-1]] = layer
            curr_layer += 1

    return model

def static_lstm_layer_settings(model, weight_bits, group_size, num_terms):
    curr_layer = 0
    stats = []
    for _, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.LSTM)):
            stats.append((weight_bits, group_size, num_terms))
            curr_layer += 1

    return stats

def convert_model(model, tr_params, data_bits, data_terms):
    # copy the model, since we modify it internally
    model = deepcopy(model)
    return replace_lstm_layers(model, tr_params, data_bits, data_terms)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--data', type=str, default='./lstm_models/data/wikitext-2/',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=650,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=650,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_false',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--wb', nargs='+', type=int, help='weight bits')
    parser.add_argument('--wt', nargs='+', type=int, help='weight terms')
    parser.add_argument('--db', nargs='+', type=int, help='data bits')
    parser.add_argument('--dt', nargs='+', type=int, help='data terms')
    parser.add_argument('--gs', nargs='+', type=int, help='group sizes')
    parser.add_argument('--out-file', help='Output file')

    parser.add_argument('--nhead', type=int, default=2,
                        help='the number of heads in the encoder/decoder of the transformer model')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    corpus = data.Corpus(args.data)

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data.to(device)

    eval_batch_size = 10
    test_data = batchify(corpus.test, eval_batch_size)

    ntokens = len(corpus.dictionary)
    if args.model == 'Transformer':
        model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    model = torch.load('data/lstm.pt')
    criterion = nn.NLLLoss()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    def get_batch(source, i):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def evaluate(model, data_source):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)
    

    settings = zip(args.wb, args.wt, args.db, args.dt, args.gs)

    results = {'ppls': [], 'tmacs': [], 'param_bits': []}
    for wb, wt, db, dt, gs in settings:
        # Build model
        tr_params = static_lstm_layer_settings(model, wb, gs, wt)
        qmodel = convert_model(model, tr_params, db, dt)

        # Profile
        test_loss = evaluate(qmodel, test_data)
        set_tr_tracking(qmodel, False)

        # Evaluate
        test_loss = evaluate(qmodel, test_data)
        inputs = (get_batch(test_data, 0)[0], model.init_hidden(eval_batch_size))
        tmacs, param_bits = profile_model.get_model_ops(qmodel, inputs=inputs)
        ppl = math.exp(test_loss)
        results['ppls'].append(ppl)
        results['tmacs'].append(tmacs)
        results['param_bits'].append(param_bits)
        print(wb, wt, db, dt, gs, ppl, tmacs, param_bits)

    with open(args.out_file, 'w') as fp:
        json.dump(results, fp)