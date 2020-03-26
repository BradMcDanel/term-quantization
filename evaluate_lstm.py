import argparse
import math
from copy import deepcopy

import torch
import torch.nn as nn

import word_language_model.model as model
import word_language_model.data as data
from tr_layer import TRLSTMLayer

import sys
sys.path.insert(0, './word_language_model')

def get_model_ops(model):
    def tr_lstm_ops(m, x, y):
        x = x[0]

        kernel_ops = 1

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * m.linear.in_features

        # convert to term ops
        if m.group_size == 1:
            weight_terms = min(m.num_terms, m.weight_bits)
        else:
            weight_terms = m.num_terms
        data_terms = min(m.data_terms, m.data_bits)
        alpha = weight_terms / m.group_size
        total_ops = data_terms * alpha * total_ops
        m.linear.total_ops += torch.Tensor([int(total_ops)])

    dummy_input = torch.randn(1, 1, 28, 28).cuda()
    custom_ops = {
        TRLinearLayer: tr_linear_ops,
        nn.Conv2d: thop.count_hooks.zero_ops,
        nn.BatchNorm2d: thop.count_hooks.zero_ops,
        nn.Linear: thop.count_hooks.zero_ops,
        nn.AvgPool2d: thop.count_hooks.zero_ops,
        nn.AdaptiveAvgPool2d: thop.count_hooks.zero_ops
    }

    return thop.profile(model, inputs=(dummy_input,), custom_ops=custom_ops)


def replace_lstm_layers(model, tr_params, data_bits, data_terms):
    curr_layer = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.LSTM)):
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            weight_bits, group_size, weight_terms = tr_params[curr_layer]
            layer = TRLSTMLayer(layer, data_bits, data_terms, weight_bits,
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
    parser.add_argument('--data', type=str, default='./word_language_model/data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')

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

    state_dict = torch.load('data/lstm.pt')
    model = torch.load('data/lstm.pt')
    criterion = nn.NLLLoss()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)


    # get_batch subdivides the source data into chunks of length args.bptt.
    # If source is equal to the example output of the batchify function, with
    # a bptt-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

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
        if args.model != 'Transformer':
            hidden = model.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i)
                if args.model == 'Transformer':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    output, hidden = model(data, hidden)
                    hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

    tr_params = static_lstm_layer_settings(model, 4, 4, 4)
    qmodel = convert_model(model, tr_params, 4, 4)
    test_loss = evaluate(qmodel, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)