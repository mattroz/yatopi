import torch

import torch.nn as nn
from torchvision.models import resnet18, mobilenet_v2


def init_weights(m):
    if type(m) == nn.Conv2d:
        #torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight)
        #torch.nn.init.normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class SampleClassificationNetwork(nn.Module):
    def __init__(self, nclasses=3):
        super().__init__()
        basenet = mobilenet_v2(pretrained=True)
        modules = list(basenet.children())[:-1]
        self.basenet = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.final = nn.Linear(in_features=1280, out_features=nclasses)

    def forward(self, inp):
        x = self.basenet(inp)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.final(x)
        return x
