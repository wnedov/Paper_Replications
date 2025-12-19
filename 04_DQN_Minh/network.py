import torch.nn as nn
import torch

class DQNNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.cc1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4) #20
        self.cc2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) #9
        self.cc3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) #7
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.relu(self.cc1(x))
        x = self.relu(self.cc2(x))
        x = self.relu(self.cc3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


        