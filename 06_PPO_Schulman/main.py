import argparse
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import (
    TransformReward, FrameStackObservation, AtariPreprocessing,
    TimeLimit, RecordEpisodeStatistics, ClipAction,
    NormalizeObservation, NormalizeReward, TransformObservation,
)
import ale_py

from common.networks import NatureCNN, MLP, DiscreteHead, GaussianHead, CriticHead
from common.RLWrappers import EpisodicLifeEnv
from agent import PPOAgent
from ppo import PPO

def make_atari_env(env_id):
    def thunk():
        env = gym.make(env_id, render_mode=None, frameskip=1, repeat_action_probability=0.0)
        env = AtariPreprocessing(
            env, grayscale_obs=True, scale_obs=False,
            screen_size=84, frame_skip=4, terminal_on_life_loss=True, noop_max=30,
        )
        env = RecordEpisodeStatistics(env)
        env = TransformReward(env, lambda r: np.sign(r))
        env = FrameStackObservation(env, stack_size=4)
        env = TimeLimit(env, max_episode_steps=4500)
        env = EpisodicLifeEnv(env)
        return env
    return thunk


def make_continuous_env(env_id):
    def thunk():
        env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        env = ClipAction(env)
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = NormalizeReward(env, gamma=GAMMA)
        env = TransformReward(env, lambda r: np.clip(r, -10, 10))
        return env
    return thunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO (Schulman et al. 2017)")
    parser.add_argument("--env-type", default="atari", choices=["atari", "continuous"])
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--run-name", default="ppo_run")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    is_continuous = args.env_type == "continuous"

    # Common parameters
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    EPSILON = 0.1
    C1 = 1.0
    C2 = 0.01
    MAX_GRAD_NORM = 0.5
    ANNEAL_LR = True
    MILESTONES = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]
    NUM_ENVS = 8

    if args.env_type == "atari":
        # Atari Hyperparameters
        num_steps = 128
        learning_rate = 2.5e-4
        num_epochs = 3
        minibatch_size = 256
        
        envs = gym.vector.SyncVectorEnv([make_atari_env(args.env_id) for _ in range(NUM_ENVS)])
        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.n
        feature_dim = 512
        
        actor_backbone = NatureCNN(obs_dim, feature_dim)
        critic_backbone = NatureCNN(obs_dim, feature_dim)
        actor_head = DiscreteHead(feature_dim, action_dim)
        critic_head = CriticHead(feature_dim)
    else:
        # Continuous (MuJoCo) Hyperparameters
        num_steps = 2048
        learning_rate = 3e-4
        num_epochs = 10
        minibatch_size = 64
        
        envs = gym.vector.SyncVectorEnv([make_continuous_env(args.env_id) for _ in range(NUM_ENVS)])
        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]
        feature_dim = 64
        
        actor_backbone = MLP(obs_dim, feature_dim)
        critic_backbone = MLP(obs_dim, feature_dim)
        actor_head = GaussianHead(feature_dim, action_dim)
        critic_head = CriticHead(feature_dim)

    agent = PPOAgent(
        actor_backbone=actor_backbone,
        critic_backbone=critic_backbone,
        actor_head=actor_head,
        critic_head=critic_head,
        is_continuous=is_continuous,
    ).to(device)

    ppo = PPO(
        agent=agent,
        envs=envs,
        device=device,
        num_steps=num_steps,
        learning_rate=learning_rate,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        epsilon=EPSILON,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        c1=C1,
        c2=C2,
        max_grad_norm=MAX_GRAD_NORM,
        total_timesteps=args.total_timesteps,
        anneal_lr=ANNEAL_LR,
        run_name=args.run_name,
        milestones=MILESTONES,
    )

    ppo.train()