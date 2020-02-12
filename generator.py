import torch.nn as nn
import torch.nn.functional as f
from torch import cat

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc_layer_1_1 = nn.Linear(100, 256)
        self.fc_layer_1_2 = nn.Linear(10, 256)
        self.bn_layer_1 = nn.BatchNorm1d(256)

        self.fc_layer_2 = nn.Linear(512, 512)
        self.bn_layer_2 = nn.BatchNorm1d(512)

        self.fc_layer_3 = nn.Linear(512, 1024)
        self.bn_layer_3 = nn.BatchNorm1d(1024)

        self.fc_layer_4 = nn.Linear(1024, 784)

    def forward(self, input, target):
        z = f.relu(self.bn_layer_1(self.fc_layer_1_1(input)))
        y = f.relu(self.bn_layer_1(self.fc_layer_1_2(target)))

        temp = cat([z, y], 1)

        temp = f.relu(self.bn_layer_2(self.fc_layer_2(temp)))
        temp = f.relu(self.bn_layer_3(self.fc_layer_3(temp)))
        return f.sigmoid(self.fc_layer_4(temp))
