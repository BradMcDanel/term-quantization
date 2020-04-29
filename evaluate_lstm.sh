#!/bin/bash

python evaluate_lstm.py --wb 5 6 7 8 9 --wt 5 6 7 8 9 --db 8 8 8 8 8 --dt 8 8 8 8 8 --gs 1 1 1 1 1 --out-file results/lstm-quant.json
python evaluate_lstm.py --wb 8 8 8 8 8 --wt 8 12 16 20 24 --db 8 8 8 8 8 --dt 8 8 8 8 8 --gs 8 8 8 8 8 --out-file results/lstm-tr.json