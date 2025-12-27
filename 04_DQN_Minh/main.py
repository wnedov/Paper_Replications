from network import DQNNetwork
from Replaymem import ReplayMemory
import gymnasium as gym
from gymnasium.wrappers import TransformReward, FrameStackObservation, AtariPreprocessing, TimeLimit
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ale_py
import os



def create_env():
    env = gym.make("ALE/Breakout-v5", render_mode=None, frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, screen_size=84, frame_skip=4, terminal_on_life_loss=True, noop_max=30)
    env = TransformReward(env, lambda r: np.sign(r)) #np.sign(r) to clip rewards.
    env = FrameStackObservation(env, stack_size=4)
    env = TimeLimit(env, max_episode_steps=4500)
    return env


def train_network(policy_net, target_net, batch, loss_fn, optimizer, device, writer, steps_done):
    optimizer.zero_grad()
    policy_net.train()
    states, actions , next_states , rewards, terminations = zip(*batch)

    states = np.array(states) 
    next_states = np.array(next_states)

    states = torch.tensor(states, device=device, dtype=torch.float32)
    terminations = torch.tensor(terminations, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.int64).unsqueeze(1)
    next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)

    pred = policy_net(states).gather(dim=1, index=actions).squeeze()
    with torch.no_grad():
        best_actions = policy_net(next_states).argmax(dim=1).unsqueeze(1)
        future_rewards = target_net(next_states).gather(dim=1, index=best_actions).squeeze()
    y = rewards + (1 - terminations) * 0.99 * future_rewards

    loss = loss_fn(y, pred)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()

    if steps_done % 100 == 0: 
        writer.add_scalar("Loss/train", loss.item(), steps_done)



def train_episode(env, memory, policy_net, target_net, loss_fn, optimizer, steps_done, device, writer):
    episode_over = False
    state, info = env.reset()
    state = np.array(state)
    total_reward = 0;
    while not episode_over: 

        rand = np.random.rand()
        training_steps = max(0, steps_done - 50000) 
        epsilon = max(0.1, 1 - 7e-7* training_steps) #9e-7 for true linear
        if rand < epsilon:  
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                policy_net.eval()
                actions = policy_net(torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(actions).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.array(next_state)
        total_reward += reward 
        steps_done += 1

   
        memory.push(state, action, next_state, reward, terminated)

        real_game_over = info.get("lives", 0) == 0

        if real_game_over or truncated:
            episode_over = True;
        elif terminated:
            state = next_state 
        else:
            state = next_state

        if memory.__len__() < 50000 or steps_done % 4 != 0:
            continue

        batch = memory.sample(32)
        train_network(policy_net, target_net, batch, loss_fn, optimizer, device, writer, steps_done)

        if steps_done % 10000 == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
    writer.add_scalar("Score/Episode_Reward", total_reward, steps_done)
    writer.add_scalar("Epsilon", max(0.1, 1 - 7e-7 * training_steps), steps_done)
    
    return total_reward, steps_done





if __name__ == "__main__":
    env = create_env()
    memory = ReplayMemory(capacity=1000000)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    writer = SummaryWriter("runs/breakout")

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
    loss = torch.nn.SmoothL1Loss()

    steps_done = 0
    episode_count = 0
    milestone_index = 0
    CHECKPOINT_FILE = "AAAAA" # change as neccesary
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading From {CHECKPOINT_FILE}...")
        policy_net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        steps_done = 20_000_000 # change these as neccessary
        milestone_index = 5
    
    MAX_STEPS = 50_050_000
    milestones = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
   

    progress_bar = tqdm(total=MAX_STEPS, unit="step", desc="Training")
    while steps_done < MAX_STEPS:
        
        start_steps = steps_done

        total_reward, steps_done = train_episode(env, memory, policy_net, target_net, loss, optimizer, steps_done, device, writer)
        
        steps_added = steps_done - start_steps
        episode_count += 1

        progress_bar.update(steps_added)
        progress_bar.set_postfix(
            ep=episode_count, 
            len=steps_added,
            score=total_reward 
        )

        if milestone_index < len(milestones):
            target = milestones[milestone_index]
            
            if steps_done >= target:
                filename = f"weights_{target//1_000_000}M.pth"
                torch.save(policy_net.state_dict(), filename)
                progress_bar.write(f"  Checkpoint saved: {filename}")
                milestone_index += 1

    progress_bar.close()
