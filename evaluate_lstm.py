import argparse
import time
import math
import os
import hashlib

import torch

import awd_lstm_lm.data

from awd_lstm_lm.utils import batchify, get_batch, repackage_hidden

def evaluate(model, data_source, batch_size=10):
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    
    loss = total_loss.item() / len(data_source)
    ppl = math.exp(loss)

    return ppl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('model_path', help='path to model')
    parser.add_argument('--data', type=str, default='awd_lstm_lm/data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--bptt', type=int, default=70, help='sequence length')
    parser.add_argument('--cuda', action='store_false', help='use CUDA')
    args = parser.parse_args()

    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        corpus = torch.load(fn)
    else:
        corpus = awd_lstm_lm.data.Corpus(args.data)
        torch.save(corpus, fn)

    test_batch_size = 1
    test_data = batchify(corpus.test, test_batch_size, args)

    with open(args.model_path, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    tr_params = awd_lstm_lm.static_lstm_layer_settings(model, 4, 1, 4)
    qmodel = awd_lstm_lm.convert_model(model, tr_params, 8, 8)

    ppl = evaluate(qmodel, test_data, test_batch_size)
    print(ppl)
