import torch
import torch.nn as nn
from .utils import load_state_dict_from_url

import sys 
sys.path.append('..')

import booth

__all__ = ['AlexNet', 'alexnet', 'convert_alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def convert_alexnet(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                    d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                    data_stationary):
    layers = []
    curr_layer = 0
    for i, layer in enumerate(model.features):
        if isinstance(layer, nn.Conv2d):
            if curr_layer < data_stationary: 
                layer_group_size = min(layer.weight.shape[1], w_stat_group)
                layer.weight.data = booth.booth_cuda.radix_2_mod(layer.weight.data, 2**-15,
                                                                 layer_group_size, w_stat_terms)
            else:
                layer.weight.data = booth.booth_cuda.radix_2_mod(layer.weight.data, 2**-15,
                                                                 w_move_group, w_move_terms)
        elif isinstance(layer, nn.ReLU):
            if i == len(model.features) - 1:
                pass
            elif curr_layer < data_stationary:
                layer = nn.Sequential(
                        nn.ReLU(inplace=True),
                        booth.Radix2ModGroup(2**-15, d_move_group, d_move_terms),
                    )
            else:
                layer = nn.Sequential(
                        nn.ReLU(inplace=True),
                        booth.Radix2ModGroup(2**-15, d_stat_group, d_stat_terms),
                    )

        layers.append(layer)

        if isinstance(layer, nn.Conv2d):
            curr_layer += 1

    model.features = nn.Sequential(*layers)

    return model