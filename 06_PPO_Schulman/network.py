import torch.nn as nn
import torch

class PPODiscreteNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(PPODiscreteNetwork, self).__init__()
        self.cc1 = nn.Conv2d(input_dim, 16, kernel_size=8, stride=4) #20
        self.cc2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) #9
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9*9*32, 256)
        self.actor_head = nn.Linear(256, output_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.relu(self.cc1(x))
        x = self.relu(self.cc2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        action_logits = self.actor_head(x)
        state_values = self.critic_head(x)
        return action_logits, state_values

        