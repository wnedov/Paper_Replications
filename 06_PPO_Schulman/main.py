from network import PPODiscreteNetwork
import gymnasium as gym
from gymnasium.wrappers import TransformReward, FrameStackObservation, AtariPreprocessing, TimeLimit
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import ale_py
import os
from common.RLWrappers import EpisodicLifeEnv


def create_env():
    env = gym.make("ALE/Breakout-v5", render_mode=None, frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, screen_size=84, frame_skip=4, terminal_on_life_loss=True, noop_max=30)
    env = TransformReward(env, lambda r: np.sign(r)) #np.sign(r) to clip rewards.
    env = FrameStackObservation(env, stack_size=4)
    env = TimeLimit(env, max_episode_steps=4500)
    env = EpisodicLifeEnv(env)
    return env



def GAE(next_done, next_value, memory, gamma=0.99, lam=0.95):
    
    advantages = torch.zeros_like(memory["rewards"], device=memory["rewards"].device)
    lastgae = 0.0
    for t in reversed(range(128)):
        if t == 127:
            non_terminal = 1.0 - next_done
            next_values = next_value
        else:
            non_terminal = 1.0 - memory["dones"][t + 1]
            next_values = memory["values"][t + 1]
        
        delta = memory["rewards"][t] + gamma * next_values * non_terminal - memory["values"][t]
        lastgae = delta + (gamma * lam * non_terminal * lastgae)
        advantages[t] = lastgae
    
    returns = advantages + memory["values"]
    return advantages, returns

            
def get_data(envs, policy_net, prev_state, memory, device):
    state = prev_state
    for step in range(128): 


        memory["states"][step] = torch.tensor(state, device=device, dtype=torch.float32).detach()

        with torch.no_grad():
            policy_net.eval()

            action, critic = policy_net(torch.tensor(states, device=device, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=action)
            action = dist.sample() 

        memory["actions"][step] = action.detach()
        memory["log_probs"][step] = dist.log_prob(action).detach()
        memory["values"][step] = critic.squeeze().detach()


        next_state, reward, terminated, truncated, infos = envs.step(action)
        done = np.logical_or(terminated, truncated)

        memory["rewards"][step] = torch.tensor(reward, device=device, dtype=torch.float32).detach()
        memory["dones"][step] = torch.tensor(done, device=device, dtype=torch.float32).detach()
        # need concept of total reward? I guess we also increase steps done? But there are 8 envs.. 

        if "_final_observation" in infos.keys():
            for i, is_final in enumerate(infos["_final_observation"]):
                if is_final:
                    real_terminal_obs = infos["final_observation"][i]
                    #Add to memory how?
  
        state = next_state

    with torch.no_grad():
        policy_net.eval()
        _, next_value = policy_net(torch.tensor(next_state, device=device, dtype=torch.float32))
        next_value = next_value.squeeze()
        #Again, do done flag properly here...

    return next_value, next_done, state #actually, where do i get next done from?


if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv([create_env for _ in range(8)])
    states, info = envs.reset()
    next_obs = torch.tensor(states)
    memory = {
        "states": torch.zeros((128, 8, 4, 84, 84), device=device),
        "actions": torch.zeros((128, 8), device=device),
        "log_probs": torch.zeros((128, 8), device=device),
        "rewards": torch.zeros((128, 8), device=device),
        "dones": torch.zeros((128, 8), device=device),
        "values": torch.zeros((128, 8), device=device)
    }
 
    writer = SummaryWriter("runs/breakout")

    state_dim = envs.observation_space.shape[0] 
    action_dim = envs.action_space.n
    policy_net = PPODiscreteNetwork(input_dim=state_dim, output_dim=action_dim).to(device)
    #need target net? Probably, right?

    #anneal alpha somehow? Can we change optimiser params every time we call it? 
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=2.5e-4)
    loss = ...



    steps_done = 0
    episode_count = 0
    milestone_index = 0
    CHECKPOINT_FILE = "AAAAA" # change as neccesary
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading From {CHECKPOINT_FILE}...")
        policy_net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=device))
        steps_done = 20_000_000 # change these as neccessary
        milestone_index = 5
    
    MAX_STEPS = 50_050_000
    milestones = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
   

    progress_bar = tqdm(total=MAX_STEPS, unit="step", desc="Training")
    while steps_done < MAX_STEPS:
        
        start_steps = steps_done

        memory = get_data(envs, policy_net, states, memory, device)  # Collect data from environments  





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




