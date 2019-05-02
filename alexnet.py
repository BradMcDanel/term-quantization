import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from cgm import CGM


__all__ = ['AlexNet', 'alexnet']

class AlexNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, act='relu', group_size=1):
        if act == 'relu':
            act_f = lambda: nn.ReLU(inplace=True)
        elif act == 'group-relu':
            act_f = lambda: CGM(group_size)
        else:
            raise RuntimeError('Unsupported activation function: {}'.format(act))

        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            act_f(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            act_f(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_f(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_f(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_f(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            act_f(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            act_f(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x