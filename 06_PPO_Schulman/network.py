import torch.nn as nn
import numpy as np
import torch

class PPODiscreteNetwork(nn.Module):

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def __init__(self, input_dim, output_dim):
        super(PPODiscreteNetwork, self).__init__()
        self.cc1 = self.layer_init(nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)) #20
        self.cc2 = self.layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)) #9
        self.cc3 = self.layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)) #7
        self.relu = nn.ReLU()
        self.fc1 = self.layer_init(nn.Linear(7*7*64, 512))
        self.actor_head = self.layer_init(nn.Linear(512, output_dim), std=0.01)
        self.critic_head = self.layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.relu(self.cc1(x))
        x = self.relu(self.cc2(x))
        x = self.relu(self.cc3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        action_logits = self.actor_head(x)
        state_values = self.critic_head(x)
        return action_logits, state_values

        