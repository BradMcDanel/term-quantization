# Term Revealing

A method of applying an addition quantization step onto already quantized DNNs. 

<p align="center"> 
<img src="https://github.com/BradMcDanel/term-revealing/blob/master/figures/term-reveal.png" width=700>
</p>

### Requirements:

```
matplotlib==3.2.0
numpy==1.17.0
torch==1.4.0
torchvision==0.4.2
```

## Project Layout

`cnn_models/` supports applying term revealing onto pre-trained CNNs (mainly from torchvision).

`figures/` contains figures which are generated using scripts from the `visualize` folder.

`kernels/` contains CUDA kernels which implement term revealing.

`lstm_models/` is a clone of PyTorch examples [word_language_model](https://github.com/pytorch/examples/tree/master/word_language_model).

`pretrained_models/` is used to store a pretrained MLP and LSTM model. Directions are provided to download these from google drive. Additionally, these can be trained (see below).

`results/` contains JSON files of evaluated results used in the visualization.

`thop/` is a fork of [PyTorch OpCounter](https://github.com/Lyken17/pytorch-OpCounter) with support added counting bitwise operations (i.e., number of term pair multiplications) as opposed to FLOPS.

`verilog/` provides the verilog codebase implementing the term revealing system. This is used for the hardware evaluation.

`visualize/` contains scripts to generate visualizations of term revealing. Mainly, comparsions of term revealing to conventional uniform quantization.

The remaining files at the root are used to perform the training and evaluation of term revealing.

## Training

### Training MLP
We adapt the PyTorch examples [MNIST project](https://github.com/pytorch/examples/tree/master/mnist) to train an MLP instead of a CNN. The model definition is [here](https://github.com/BradMcDanel/term-revealing/blob/master/train_mlp.py#L10).

To train the MLP for MNIST, run `python train_mlp.py`, which saves the model to `pretrained_models/mnist_mlp.pt`.

### Training LSTM
We directly use the PyTorch examples [word_language_model project](https://github.com/pytorch/examples/tree/master/word_language_model) to train an LSTM.

To train the LSTM for Wikitext-2, navigate to the `lstm_models` folder and run `python train_lstm.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied`, which saves the model to `pretrained_models/lstm.pt`.

### Training CNNs
No training is required for CNNs as available pre-trained models are used. For VGG-16, ResNet-18, and MobileNet-V2, we use [torchvision](https://github.com/pytorch/vision). For EfficientNet-b0, we use [Luke Melas-Kyriazi's pytorch implementation](https://github.com/lukemelas/EfficientNet-PyTorch).


## Evaluation


### Evaluate MLP
To apply multiple uniform Quantization (QT) and Term Revealing (TR) on the pretrained MLP for MNIST, run `bash evaluate_mlp.sh`, which calls the `evaluate_mlp.py` script for multiple settings. The results are saved in JSON format in the `results` folder as `results/mnist-quant.json` and `results/mnist-tr.json`. These are used by `visualize` scripts discussed later.


### Evaluate LSTM
Same as with the MLP. Run `bash evaluate_lstm.sh`, which calls the `evaluate_lstm.py` script for QT and TR. The results are saved in the `results` folder as `results/lstm-quant.json` and `results/lstm-tr.json`.


### Evaluate CNNs
First, open `evaluate_cnn.sh` and modify the first argument (currently `/hdd1/datasets/`) to your folder containing an `imagenet` folder. This `imagenet` folder contains a `val` folder which was generated using the `valprep.sh` script in the [Pytorch ImageNet repository](https://github.com/pytorch/examples/tree/master/imagenet).

Run `bash evaluate_cnn.sh`, which calls the `evaluate_cnn.py` script for QT and TR across VGG-16, ResNet-18, MobileNet-V2, and EfficientNet-b0. The results are saved in the `results` folder. For the CNNs the QT and TR settings are saved in a single JSON file (e.g., `resnet18-results.json`). Note that this takes multiple hours to run due to the number of settings evaluated across the four models.

Additionally, we evaluate the impact of the group size (g) and group budget (k) parameters on ResNet-18. Run `python evaluate_group_size.py <dataset root> -a resnet18` to perform a grid search across group size and group budget. The results are saved to `results/resnet18-group-size-results.json`.


## Visualization

### Quantization versus Term Revealing

To visualize the results across the MLP, LSTM, and CNNs comparing QT and TR, run `python visualize/quant_vs_term_reveal`, from the root folder. This will generate the comparison as `figures/tr-comp.pdf`.

### Impact of Group Size and Group Budget

To plot the group size and group budget settings for resnet18, run `python visualize/group_size.py`, from the root folder. This will generate the comparison as `figures/group-size-accuracy.pdf`.

### Layerwise Quantizaton Error

The layerwise quantization error of ResNet-18 for various QT and TR settings can be generating by running `python visualize/quant_error.py -a resnet18`, which will produce `figures/quant_error_analysis.pdf`. Note that this script loads the pre-trained model to perform the quantization right before it is visualized (thus taking a few minutes).


### FPGA Evaluation

The visualization of the FPGA comparison can be generated by running `python visualize/fpga_results.py`, which outputs `figures/fpga-models.pdf`. For more details on the codebase used to generate the latency and energy efficiency numbers used for this evaluation, see the [verilog](https://github.com/BradMcDanel/term-revealing/tree/master/verilog) folder in this repository.
