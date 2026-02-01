from network import PPODiscreteNetwork
import gymnasium as gym
from gymnasium.wrappers import TransformReward, FrameStackObservation, AtariPreprocessing, TimeLimit, RecordEpisodeStatistics
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
    env = RecordEpisodeStatistics(env)
    env = TransformReward(env, lambda r: np.sign(r)) #np.sign(r) to clip rewards.
    env = FrameStackObservation(env, stack_size=4)
    env = TimeLimit(env, max_episode_steps=4500)
    env = EpisodicLifeEnv(env)
    return env


def train_policy(policy_net, memory, optimiser, device, advantages, returns, epsilon, c1=1, c2=0.01):

    batch_size = 1024
    minibatch_size = 256

    optimiser.zero_grad()
    policy_net.train()

    states = memory["states"].flatten(start_dim=0, end_dim=1).to(device)
    actions = memory["actions"].flatten(start_dim=0, end_dim=1).to(device)
    old_log_probs = memory["log_probs"].flatten(start_dim=0, end_dim=1).to(device)

    advantages = advantages.flatten(0, 1).to(device)
    advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = returns.flatten(0, 1).to(device)

    total_losses = []
    actor_losses = []
    critic_losses = []
    entropy_losses = []

    for epoch in range(3):

        inds = torch.randperm(batch_size, device=device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = inds[start:end]
            mb_states = states[mb_inds]
            mb_actions = actions[mb_inds]
            mb_log_probs = old_log_probs[mb_inds]
            mb_advantages = advantages_norm[mb_inds]
            mb_returns = returns[mb_inds]


            policy, values = policy_net(mb_states) 
            values = values.squeeze()
            dist = torch.distributions.Categorical(logits=policy)

            r = torch.exp(dist.log_prob(mb_actions) - mb_log_probs) 
            clipped_loss = torch.min(r*mb_advantages, torch.clamp(r, 1 - epsilon, 1 + epsilon) * mb_advantages).mean()
            critic_loss = torch.nn.MSELoss() 
            entropy_loss = dist.entropy().mean()

            loss = -clipped_loss + c1 * critic_loss(values, mb_returns) - c2 * entropy_loss 
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5) 
            loss.backward() 
            optimiser.step()

            total_losses.append(loss.item())
            actor_losses.append(clipped_loss.item())
            critic_losses.append(critic_loss(values, mb_returns).item())
            entropy_losses.append(entropy_loss.item())

    return np.mean(total_losses), np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)


def GAE(next_done, next_value, memory, gamma=0.99, lam=0.95):
    
    advantages = torch.zeros_like(memory["rewards"], device=memory["rewards"].device)
    lastgae = 0.0
    for t in reversed(range(128)):
        if t == 127:
            non_terminal = 1.0 - next_done
            next_values = next_value
        else:
            non_terminal = 1.0 - memory["dones"][t]
            next_values = memory["values"][t + 1]
        
        delta = memory["rewards"][t] + gamma * next_values * non_terminal - memory["values"][t]
        lastgae = delta + (gamma * lam * non_terminal * lastgae)
        advantages[t] = lastgae
    
    returns = advantages + memory["values"]
    return advantages, returns

            
def get_data(envs, policy_net, prev_state, memory, device, steps_done, writer):
    ep_returns = []
    state = prev_state
    for step in range(128): 


        memory["states"][step] = torch.tensor(state, device=device, dtype=torch.float32).detach()

        with torch.no_grad():
            policy_net.eval()

            action, critic = policy_net(torch.tensor(state, device=device, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=action)
            action = dist.sample() 

        memory["actions"][step] = action.detach()
        memory["log_probs"][step] = dist.log_prob(action).detach()
        memory["values"][step] = critic.squeeze().detach()


        next_state, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        if step == 127:
            next_done = torch.tensor(done, device=device, dtype=torch.float32).detach()

        memory["rewards"][step] = torch.tensor(reward, device=device, dtype=torch.float32).detach()
        memory["dones"][step] = torch.tensor(done, device=device, dtype=torch.float32).detach()


        if "episode" in infos:
            if "_episode" in infos:
                for i in range(len(infos["_episode"])):
                    if infos["_episode"][i]: 
                        r = infos["episode"]["r"][i]
                        l = infos["episode"]["l"][i]
                        
                        ep_returns.append(r)
                        writer.add_scalar("charts/episodic_return", r, steps_done)
                        writer.add_scalar("charts/episodic_length", l, steps_done)
  
        state = next_state

    with torch.no_grad():
        policy_net.eval()
        _, next_value = policy_net(torch.tensor(next_state, device=device, dtype=torch.float32))
        next_value = next_value.squeeze()
        
    mean_return = 0 if len(ep_returns) == 0 else np.mean(ep_returns)
    return next_value, next_done, state, mean_return


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

    state_dim = envs.single_observation_space.shape[0] 
    action_dim = envs.single_action_space.n
    policy_net = PPODiscreteNetwork(input_dim=state_dim, output_dim=action_dim).to(device)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=2.5e-4, eps=1e-5)

    steps_done = 0
    milestone_index = 0

    epsilon = 0.1
    c1 = 1.0
    c2 = 0.01

    CHECKPOINT_FILE = "AAAAA" # change as neccesary
    
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Loading From {CHECKPOINT_FILE}...")
        policy_net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=device))
        steps_done = 20_000_000 # change these as neccessary
        milestone_index = 5
    
    MAX_STEPS = 20_050_000
    milestones = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000]
   

    progress_bar = tqdm(total=MAX_STEPS, unit="step", desc="Training")
    while steps_done < MAX_STEPS:
        
        start_steps = steps_done

        next_value, next_done, next_state, ep_returns = get_data(envs, policy_net, states, memory, device, steps_done, writer)  
        steps_done += 128 * 8  
        states = next_state 
        advantages, returns = GAE(next_done, next_value, memory)
        loss, pg_loss, v_loss, ent_loss = train_policy(policy_net, memory, optimizer, device, advantages, returns, epsilon, c1, c2);

        steps_added = steps_done - start_steps

        frac = 1.0 - (steps_done / MAX_STEPS)
        lrnow = 2.5e-4 * frac
        epsilon = 0.1 * frac
        optimizer.param_groups[0]["lr"] = lrnow 

        writer.add_scalar("losses/total_loss", loss, steps_done)
        writer.add_scalar("losses/policy_loss", pg_loss, steps_done)
        writer.add_scalar("losses/value_loss", v_loss, steps_done)
        writer.add_scalar("losses/entropy", ent_loss, steps_done)

        progress_bar.update(steps_added)
        progress_bar.set_postfix(
            ep_return=ep_returns,
            loss=f"{loss:.4f}"
        )
        if milestone_index < len(milestones):
            target = milestones[milestone_index]
            
            if steps_done >= target:
                filename = f"weights_{target//1_000_000}M.pth"
                torch.save(policy_net.state_dict(), filename)
                progress_bar.write(f"  Checkpoint saved: {filename}")
                milestone_index += 1

    progress_bar.close()