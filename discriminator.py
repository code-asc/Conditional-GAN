import torch.nn as nn
import torch.nn.functional as f
import torch.cat as cat

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc_layer_1_1 = nn.Linear(784, 1024)
        self.fc_layer_1_2 = nn.Linear(10, 1024)

        self.fc_layer_2 = nn.Linear(2048, 512)
        self.bn_layer_2 = nn.BatchNorm1d(512)

        self.fc_layer_3 = nn.Linear(512, 256)
        self.bn_layer_3 = nn.BatchNorm1d(256)

        self.fc_layer_4 = nn.Linear(256, 1)

    def forward(self, input, target):
        z = f.leaky_relu(self.fc_layer_1_1(input), 0.2)
        y = f.leaky_relu(self.fc_layer_1_2(target), 0.2)

        temp = cat([z, y], 1)

        temp = f.leaky_relu(self.bn_layer_2(self.fc_layer_2(temp)), 0.2)
        temp = f.leaky_relu(self.bn_layer_3(self.fc_layer_3(temp)), 0.2)

        return f.sigmoid(self.fc_layer_4(temp))
