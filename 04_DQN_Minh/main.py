from network import DQNNetwork
from Replaymem import ReplayMemory
import gymnasium as gym
from gymnasium.wrappers import TransformReward, FrameStackObservation, AtariPreprocessing
import numpy as np
import torch


#use apple GPU here...., somehow


def create_env():
    #Okay, i've defined terminal state on life loss here, but what about truncated state? - Also, I swear i needed to scale here.
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, screen_size=84, frame_skip=4, terminal_on_life_loss=True)
    env = TransformReward(env, lambda r: np.sign(r))
    env = FrameStackObservation(env, num_stack=4)
    return env


def train_network(policy_net, batch, loss_fn, optimizer):


    policy_net.train()


    # if episode_over:
    #     y = reward
    # else:
    #     with torch.no_grad():
    #         target_net.eval()
    #         next_q_values = target_net(torch.tensor(next_state))
    #         y = reward + 0.99 * torch.max(next_q_values).item()

    loss = loss_fn(transition.action, torch.tensor(y)) #Do i still need inference here...? 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def train_episode(env, memory, policy_net, target_net, loss_fn, optimizer):
    episode_over = False
    state, info = env.reset()
    steps_done = 0
    while not episode_over: 
        rand = np.random.rand()
        epsilon = 
        if rand < 0.1:  # epsilon-greedy(how to anneal?)
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                policy_net.eval()
                actions = policy_net(torch.tensor(state))
                action = torch.argmax(actions).item()
        next_state, reward, terminated, truncated, info = env.step(action)
        steps_done += 1
        episode_over = terminated or truncated
        memory.push(state, action, next_state, reward, episode_over)
        state = next_state

        if memory.__len__() < 32:
            continue

        batch = memory.sample(32)
        train_network(policy_net, batch, loss_fn, optimizer)

        if steps_done % 10000:
            target_net.load_state_dict()





if __name__ == "__main__":
    env = create_env()
    memory = ReplayMemory(capacity=100000)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim)
    target_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.RMSprop(policy_net.parameters(),
                                    lr=0.00025, 
                                    alpha=0.95, 
                                    eps=0.01, 
                                    momentum=0.95)
    loss = torch.nn.MSELoss()
    optimizer.zero_grad()
    train_episode(env, memory, policy_net, target_net)
