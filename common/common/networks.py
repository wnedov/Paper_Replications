import torch.nn as nn
import numpy as np
import torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, use_init=True):
        if not use_init: 
            return layer
        else: 
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer

class NatureCNN(nn.Module):

    def __init__(self, input_dim, feature_dim=512, use_init=True):
        super().__init__()
        
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(input_dim, 32, kernel_size=8, stride=4), use_init=use_init),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), use_init=use_init),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1), use_init=use_init),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            layer_init(nn.Linear(64 * 7 * 7, feature_dim), std=np.sqrt(2), use_init=use_init),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.float() / 255
        return self.cnn(x)
    
class MLP(nn.Module):

    def __init__(self, input_dim, feature_dim=64, use_init=True):
        super().__init__()
        
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(input_dim, feature_dim), std=np.sqrt(2), use_init=use_init),
            nn.Tanh(),
            layer_init(nn.Linear(feature_dim, feature_dim), std=np.sqrt(2), use_init=use_init),
            nn.Tanh()
        )

    def forward(self, x):
        return self.mlp(x)


class CriticHead(nn.Module): 

    def __init__(self, feature_dim, use_init=True):
        super().__init__()
        self.critic = layer_init(nn.Linear(feature_dim, 1), std=1.0, use_init=use_init)
    
    def forward(self, x): 
        val = self.critic(x)
        return val
    

class DiscreteHead(nn.Module):

    def __init__(self, feature_dim, action_dim, use_init=True):
        super().__init__()
        self.actor = layer_init(nn.Linear(feature_dim, action_dim), std=0.01, use_init=use_init)

    def forward(self, x):
        x = self.actor(x)
        return x

class GaussianHead(nn.Module):
    def __init__(self, feature_dim, action_dim, use_init=True):
        super().__init__()
        self.mean_layer = layer_init(nn.Linear(feature_dim, action_dim), std=0.01, use_init=use_init)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std) 
        return mean, std
