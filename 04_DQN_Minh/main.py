from network import DQNNetwork
from Replaymem import ReplayMemory
import gymnasium as gym
from gymnasium.wrappers import TransformReward, FrameStackObservation, AtariPreprocessing, TimeLimit
import numpy as np
import torch



def create_env():
    env = gym.make("ALE/Breakout-v5", render_mode="None")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, screen_size=84, frame_skip=4, terminal_on_life_loss=True)
    env = TransformReward(env, lambda r: np.sign(r))
    env = FrameStackObservation(env, num_stack=4)
    env = TimeLimit(env, max_episode_steps=4500)
    return env


def train_network(policy_net, target_net, batch, loss_fn, optimizer, device):
    optimizer.zero_grad()
    policy_net.train()
    states, actions , next_states , rewards, terminations = zip(*batch)

    states = torch.tensor(states, device=device)
    terminations = torch.tensor(terminations, device=device)
    actions = torch.tensor(actions, device=device).unsqueeze(1)
    next_states = torch.tensor(next_states, device=device)
    rewards = torch.tensor(rewards, device=device)

    pred = policy_net(states).gather(dim=1, index=actions).squeeze()
    future_rewards = target_net(next_states).max(dim=1)[0].detach()
    y = rewards + (1 - terminations) * 0.99 * future_rewards

    loss = loss_fn(y, pred)
    loss.backward()
    optimizer.step()



def train_episode(env, memory, policy_net, target_net, loss_fn, optimizer, steps_done, device):
    episode_over = False
    state, info = env.reset()
    while not episode_over: 
        rand = np.random.rand()
        epsilon = max(0.1, 1 - 9e-7* steps_done)
        if rand < epsilon:  
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                policy_net.eval()
                actions = policy_net(torch.tensor(state, device=device).unsqueeze(0))
                action = torch.argmax(actions).item()
        next_state, reward, terminated, truncated, info = env.step(action)
        steps_done += 1
        episode_over = terminated or truncated
        memory.push(state, action, next_state, reward, episode_over)
        state = next_state

        if memory.__len__() < 32:
            continue

        batch = memory.sample(32)
        train_network(policy_net, target_net, batch, loss_fn, optimizer, device)

        if steps_done % 10000 == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    return steps_done





if __name__ == "__main__":
    env = create_env()
    memory = ReplayMemory(capacity=1000000)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    state_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.n
    policy_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim).to(device)
    target_net = DQNNetwork(input_dim=state_dim, output_dim=action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())


    optimizer = torch.optim.RMSprop(policy_net.parameters(),
                                    lr=0.00025, 
                                    alpha=0.95, 
                                    eps=0.01, 
                                    momentum=0.95)
    loss = torch.nn.MSELoss()

    steps_done = 0
    for i in range(5000):
        steps_done = train_episode(env, memory, policy_net, target_net, loss, optimizer, steps_done, device)
