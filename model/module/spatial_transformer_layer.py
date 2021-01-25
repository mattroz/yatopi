import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):

    def __init__(self):
        super().__init__()
        self.localization_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 16, kernel_size=(1,8), stride=(1,3)),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=8, stride=8), #nn.Conv2d(16, 16, kernel_size=6, stride=6) for (N,1,32,100) input
            nn.ReLU(True),
        )
        self.regressor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )
        self.regressor[2].weight.data.zero_()
        self.regressor[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, input_tensor):
        x = self.localization_net(input_tensor)
        x = x.view(-1, 16)
        theta = self.regressor(x)
        theta = theta.view(-1, 2, 3)
        #print(theta)
        grid = F.affine_grid(theta, input_tensor.size())
        x = F.grid_sample(input_tensor, grid)

        return x