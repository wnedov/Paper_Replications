import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class PPO:

    def __init__(self, agent, envs, device,
                 num_steps=128, learning_rate=2.5e-4,
                 gamma=0.99, gae_lambda=0.95, epsilon=0.1,
                 num_epochs=3, minibatch_size=256,
                 c1=1.0, c2=0.01, max_grad_norm=0.5,
                 total_timesteps=10_000_000, anneal_lr=True,
                 run_name="ppo_run", milestones=None):

        self.agent = agent
        self.envs = envs
        self.device = device
        self.num_steps = num_steps
        self.num_envs = envs.num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.num_epochs = num_epochs
        self.batch_size = num_steps * self.num_envs
        self.minibatch_size = minibatch_size
        self.c1 = c1
        self.c2 = c2
        self.max_grad_norm = max_grad_norm
        self.total_timesteps = total_timesteps
        self.anneal_lr = anneal_lr
        self.run_name = run_name
        self.milestones = milestones or []

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
        self.writer = SummaryWriter(f"runs/{run_name}")

    def _init_memory(self):
        obs_shape = self.envs.single_observation_space.shape
        act_shape = self.envs.single_action_space.shape

        memory = {
            "states": torch.zeros((self.num_steps, self.num_envs, *obs_shape), device=self.device),
            "actions": torch.zeros((self.num_steps, self.num_envs, *act_shape), device=self.device),
            "log_probs": torch.zeros((self.num_steps, self.num_envs), device=self.device),
            "rewards": torch.zeros((self.num_steps, self.num_envs), device=self.device),
            "dones": torch.zeros((self.num_steps, self.num_envs), device=self.device),
            "values": torch.zeros((self.num_steps, self.num_envs), device=self.device),
        }
        return memory

    def compute_gae(self, next_value, next_done, memory):
        advantages = torch.zeros_like(memory["rewards"])
        lastgae = 0.0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                non_terminal = 1.0 - memory["dones"][t]
                next_values = memory["values"][t + 1]

            delta = memory["rewards"][t] + self.gamma * next_values * non_terminal - memory["values"][t]
            lastgae = delta + self.gamma * self.gae_lambda * non_terminal * lastgae
            advantages[t] = lastgae

        returns = advantages + memory["values"]
        return advantages, returns

    def collect_rollouts(self, states, memory, steps_done):
        ep_returns = []
        state = states
        next_done = None

        for step in range(self.num_steps):
            memory["states"][step] = torch.tensor(state, device=self.device, dtype=torch.float32)

            with torch.no_grad():
                self.agent.eval()
                state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32)
                action, log_prob, _, value = self.agent.get_action_and_value(state_tensor)

            memory["actions"][step] = action.detach()
            memory["log_probs"][step] = log_prob.detach()
            memory["values"][step] = value.squeeze().detach()

            next_state, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            if step == self.num_steps - 1:
                next_done = torch.tensor(done, device=self.device, dtype=torch.float32)

            memory["rewards"][step] = torch.tensor(reward, device=self.device, dtype=torch.float32)
            memory["dones"][step] = torch.tensor(done, device=self.device, dtype=torch.float32)

            if "episode" in infos and "_episode" in infos:
                for i in range(len(infos["_episode"])):
                    if infos["_episode"][i]:
                        r = infos["episode"]["r"][i]
                        l = infos["episode"]["l"][i]
                        ep_returns.append(r)
                        self.writer.add_scalar("charts/episodic_return", r, steps_done)
                        self.writer.add_scalar("charts/episodic_length", l, steps_done)

            state = next_state

        with torch.no_grad():
            self.agent.eval()
            next_value = self.agent.get_value(
                torch.tensor(state, device=self.device, dtype=torch.float32)
            ).squeeze()

        mean_return = 0 if len(ep_returns) == 0 else np.mean(ep_returns)
        return next_value, next_done, state, mean_return

    def update(self, memory, advantages, returns):
        self.agent.train()

        states = memory["states"].flatten(start_dim=0, end_dim=1).to(self.device)
        actions = memory["actions"].flatten(start_dim=0, end_dim=1).to(self.device)
        old_log_probs = memory["log_probs"].flatten(start_dim=0, end_dim=1).to(self.device)

        advantages_flat = advantages.flatten(0, 1).to(self.device)
        advantages_norm = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        returns_flat = returns.flatten(0, 1).to(self.device)

        total_losses, actor_losses, critic_losses, entropy_losses = [], [], [], []

        for epoch in range(self.num_epochs):
            inds = torch.randperm(self.batch_size, device=self.device)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = inds[start:end]

                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_log_probs = old_log_probs[mb_inds]
                mb_advantages = advantages_norm[mb_inds]
                mb_returns = returns_flat[mb_inds]

                self.optimizer.zero_grad()
                _, new_log_prob, entropy, values = self.agent.get_action_and_value(mb_states, action=mb_actions)
                values = values.squeeze()

                ratio = torch.exp(new_log_prob - mb_log_probs)
                clipped_loss = torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * mb_advantages
                ).mean()

                critic_loss_fn = torch.nn.MSELoss()
                v_loss = critic_loss_fn(values, mb_returns)
                entropy_loss = entropy.mean()

                loss = -clipped_loss + self.c1 * v_loss - self.c2 * entropy_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_losses.append(loss.item())
                actor_losses.append(clipped_loss.item())
                critic_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

        return np.mean(total_losses), np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)

    def train(self):
        memory = self._init_memory()
        states, _ = self.envs.reset()
        steps_done = 0
        milestone_index = 0

        progress_bar = tqdm(total=self.total_timesteps, unit="step", desc="Training")

        while steps_done < self.total_timesteps:
            start_steps = steps_done

            next_value, next_done, states, mean_return = self.collect_rollouts(
                states, memory, steps_done
            )
            steps_done += self.num_steps * self.num_envs

            advantages, returns = self.compute_gae(next_value, next_done, memory)
            loss, pg_loss, v_loss, ent_loss = self.update(memory, advantages, returns)

            if self.anneal_lr:
                frac = 1.0 - (steps_done / self.total_timesteps)
                self.optimizer.param_groups[0]["lr"] = self.learning_rate * frac
                self.epsilon = self.initial_epsilon * frac

            self.writer.add_scalar("losses/total_loss", loss, steps_done)
            self.writer.add_scalar("losses/policy_loss", pg_loss, steps_done)
            self.writer.add_scalar("losses/value_loss", v_loss, steps_done)
            self.writer.add_scalar("losses/entropy", ent_loss, steps_done)

            steps_added = steps_done - start_steps
            progress_bar.update(steps_added)
            progress_bar.set_postfix(ep_return=mean_return, loss=f"{loss:.4f}")

            if milestone_index < len(self.milestones):
                target = self.milestones[milestone_index]
                if steps_done >= target:
                    filename = f"weights_{target // 1_000_000}M.pth"
                    torch.save(self.agent.state_dict(), filename)
                    progress_bar.write(f"  Checkpoint saved: {filename}")
                    milestone_index += 1

        progress_bar.close()
        self.writer.close()
