#!/bin/bash

python evaluate_mlp.py --wb 2 3 4 5 6 --wt 2 3 4 5 6 --db 6 6 6 6 6 --dt 6 6 6 6 6 --gs 1 1 1 1 1 --out-file results/mnist-quant.json
python evaluate_mlp.py --wb 4 4 4 4 4 --wt 6 8 10 12 14 --db 6 6 6 6 6 --dt 6 6 6 6 6 --gs 16 16 16 16 16 --out-file results/mnist-tr.json