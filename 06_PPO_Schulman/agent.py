import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal 

class PPOAgent(nn.Module):
    def __init__(self, actor_backbone, critic_backbone, actor_head, critic_head, is_continuous=False):
        super().__init__()
        self.actor_backbone = actor_backbone
        self.critic_backbone = critic_backbone
        self.actor = actor_head
        self.critic = critic_head
        self.is_continuous = is_continuous 
        
    def get_value(self, x):
        critic_features = self.critic_backbone(x)
        return self.critic(critic_features)
        
    def get_action_and_value(self, x, action=None):
        actor_features = self.actor_backbone(x)
        critic_features = self.critic_backbone(x)
        
        if self.is_continuous:
            action_mean, action_std = self.actor(actor_features)
            probs = Normal(action_mean, action_std)
        else:
            logits = self.actor(actor_features)
            probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        if self.is_continuous:
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_features)
        else:
            return action, probs.log_prob(action), probs.entropy(), self.critic(critic_features)
    
    def forward(self, x): 
        x = self.actor_backbone(x) 
        action_logits = self.actor_head(x)
        x = self.critic_backbone(x)
        state_values = self.critic_head(x)
        return action_logits, state_values