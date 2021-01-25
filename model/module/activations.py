import torch
import torch.nn as nn
import torch.nn.functional as F


def custom_softplus(x, threshold=20):
    # _x = torch.min(x, torch.full_like(x, threshold))
    # return torch.log(1 + torch.exp(_x))
    return torch.log(1 + torch.exp(-torch.abs(x))) + torch.min(x, torch.full_like(x, threshold))

class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return torch.mul(x, torch.tanh(custom_softplus(x)))
        # return torch.mul(x, torch.tanh(nn.functional.softplus(x)))
        return torch.mul(x, torch.tanh(torch.log(1 + torch.exp(x))))

class AlphaMish(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros((in_features, 1, 1)))
        self.alpha.requires_grad = True
    def forward(self, x):
        return torch.mul(x, torch.tanh(torch.mul(1 + torch.nn.functional.softplus(self.alpha), torch.nn.functional.softplus(x))))


class Swish(nn.Module):

    def __init__(self, hard=False):
        super().__init__()
        self.hard = hard

    def forward(self, input):
        if self.hard:
            return torch.mul(input, torch.div(F.relu6(torch.add(input, 3)), 6))

        return torch.mul(input, F.sigmoid(input))
