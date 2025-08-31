import time
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.envs.classic_control import PendulumEnv
from gymnasium.wrappers import RecordVideo, RescaleAction, TimeLimit
from torch.distributions import Normal


def reparametrize(mean_logstd, deterministic=True, LOG_STD_MIN=-20, LOG_STD_MAX=2):
    mean, log_std = mean_logstd.chunk(2, dim=-1)
    log_std = log_std.tanh()
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    normal = Normal(mean, log_std.exp())
    if deterministic:
        action = mean
    else:
        action = normal.rsample()
    log_prob = normal.log_prob(action)
    action = action.tanh()
    log_prob = log_prob - (1 - action.pow(2) + 1e-6).log()
    return action, log_prob


def make_mlp(input_size=3, hidden_size=256, output_size=1):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )


class ReplayBuffer(nn.Module):
    def __init__(self, min_size, max_size):
        super().__init__()
        self.reset(min_size, max_size)

    def put(self, state, action, reward, next_state):
        index = self.entries % self.max_size
        row = torch.cat([state, action, reward, next_state])
        self.sars[index] = row
        self.entries += 1

    def sample(self, n) -> torch.Tensor:
        indices = torch.randint(0, min(self.entries, self.max_size), (n, ))
        buffer = self.sars[indices].clone()
        return buffer.detach().split([3, 1, 1, 3], dim=1)

    def has_enough_data(self):
        return self.entries >= self.min_size

    def reset(self, min_size, max_size):
        self.sars = nn.Parameter(torch.zeros((max_size, 3+1+1+3), dtype=torch.float), requires_grad=False)
        self.min_size = min_size
        self.max_size = max_size
        self.entries = 0


class Agent(nn.Module):
    def __init__(self):
        super().__init__()

        self.state_dim = 3  # [cos(theta), sin(theta), theta_dot]
        self.action_dim = 1  # [torque] in[-1,1]
        self.batch_size = 200
        self.memory = ReplayBuffer(1000 * 2, 10000)

        self.actor = make_mlp(input_size=self.state_dim, output_size=self.action_dim * 2)
        self.q1 = make_mlp(input_size=self.state_dim + self.action_dim, output_size=1)
        self.q1_target = make_mlp(input_size=self.state_dim + self.action_dim, output_size=1)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2 = make_mlp(input_size=self.state_dim + self.action_dim, output_size=1)
        self.q2_target = make_mlp(input_size=self.state_dim + self.action_dim, output_size=1)
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.td_discount = 0.99
        self.ewma_forget = 0.05
        self.alpha_log = nn.Parameter(torch.tensor(np.log(1)), requires_grad=True)

        self.alphas = []
        self.alpha_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.q_gaps = []

        self.learning_steps = 0
        self.optimizer = optim.Adam(self.parameter_groups())

    def parameter_groups(self):
        return [
            {"params": self.actor.parameters(), "lr": 1e-3},
            {"params": self.q1.parameters(), "lr": 1e-3},
            {"params": self.q2.parameters(), "lr": 1e-3},
            {"params": [self.alpha_log], "lr": 6e-4}
        ]

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if not self.learning_steps:
            return torch.rand(1) * 2 - 1

        action, _ = reparametrize(self.actor(state), deterministic=self.training)
        return action

    def learn(self, state, action, reward, next_state):
        if not self.training:
            return

        self.memory.put(state, action, reward, next_state)
        if not self.memory.has_enough_data():
            return

        self.learning_steps += 1

        s_batch, a_batch, r_batch, s_prime_batch = self.memory.sample(self.batch_size)
        self.update_critic(s_batch, a_batch, r_batch, s_prime_batch)

        if self.learning_steps % 3 == 0:
            for _ in range(3):
                self.update_actor(s_batch)

            self.ema_update(input_net=self.q1, target_net=self.q1_target, ewma_forget=self.ewma_forget)
            self.ema_update(input_net=self.q2, target_net=self.q2_target, ewma_forget=self.ewma_forget)

    def update_critic(self, s_batch, a_batch, r_batch, s_prime_batch):
        with torch.no_grad():
            action, log_prob = reparametrize(self.actor(s_prime_batch), deterministic=False)
            spa = torch.cat([s_prime_batch, action], dim=-1)
            td_target = torch.min(self.q1_target(spa), self.q2_target(spa)) - self.alpha_log.exp() * log_prob
            target_q_value = r_batch + self.td_discount * td_target

        sa = torch.cat([s_batch, a_batch], dim=-1)
        q1_pred = self.q1(sa)
        q2_pred = self.q2(sa)
        q_loss = F.mse_loss(q1_pred, target_q_value) + F.mse_loss(q2_pred, target_q_value)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        self.critic_losses.append(q_loss.item())
        # Track Q1-Q2 gap as a diagnostic signal
        with torch.no_grad():
            self.q_gaps.append(torch.abs(q1_pred - q2_pred).mean().item())

    def update_actor(self, s_batch):
        action, log_prob = reparametrize(self.actor(s_batch), deterministic=False)
        sa = torch.cat([s_batch, action], dim=-1)
        actor_loss = (self.alpha_log.exp().detach() * log_prob - torch.min(self.q1(sa), self.q2(sa))).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        # only update the actor
        for param in self.q1.parameters():
            param.grad = None
        for param in self.q2.parameters():
            param.grad = None
        self.optimizer.step()
        self.actor_losses.append(actor_loss.item())

        alpha_loss = -(- self.alpha_log.exp() * (log_prob.detach() + 1)).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.alpha_log.clamp_(min=np.log(0.01), max=np.log(1))
            # Track temperature alpha
            self.alphas.append(self.alpha_log.exp().item())
        self.alpha_losses.append(alpha_loss.item())

    def ema_update(self, input_net: nn.Module, target_net: nn.Module, ewma_forget: float):
        for param_target, param_input in zip(target_net.parameters(), input_net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - ewma_forget) + param_input.data * ewma_forget)

class NoisyPendulum(PendulumEnv):
    def __init__(self, target_angle: float = 0, g: float = 10.0, eps: float = 0.0, *args, **kwargs):
        super().__init__(g=g, *args, **kwargs)
        self.eps = eps
        self.target_angle = self.angle_normalize(target_angle)
        self.max_speed = 8
        self.max_torque = 2.0 # you want more torque for weird angles

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(self, th, thdot, u):
        cost = np.abs(self.target_angle - self.angle_normalize(th))**2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return -cost

    def step(self, u):
        u = u * self.max_torque # rescale the action from -1..1 to -max_torque..max_torque
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), self.reward(th, thdot, u), False, False, {}

    def reset(self, *, seed=0, **kwargs):
        super().reset(seed=seed, **kwargs)
        eps = self.eps
        high = np.asarray([np.pi + eps, eps])
        low = np.asarray([np.pi - eps, -eps])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}


def make_env(target_angle: float = 0, g=10.0, train=True):
    eps = 0.1 if train else 0.0
    env = TimeLimit(RescaleAction(NoisyPendulum(target_angle=target_angle, render_mode='rgb_array', g=g, eps=eps),
                                  min_action=-1, max_action=1), max_episode_steps=200)
    return env


def run_episode(env, agent, record=False) -> tuple[float, int, float, float]:
    device = next(agent.parameters()).device
    state, _ = env.reset()
    state = torch.from_numpy(state).float().to(device)
    episode_return, truncated = 0.0, False
    steps = 0
    actions = []

    while not truncated:
        with torch.inference_mode():
            action = agent(state)
        a = action.item()
        next_state, reward, _, truncated, _ = env.step(a)
        next_state = torch.from_numpy(next_state).float().to(device)
        reward = torch.tensor([reward], dtype=torch.float).to(device)

        agent.learn(state, action, reward, next_state)

        episode_return += reward
        state = next_state
        steps += 1
        actions.append(a)

    action_mean = float(np.mean(actions)) if actions else 0.0
    action_std = float(np.std(actions)) if actions else 0.0
    return episode_return.item(), steps, action_mean, action_std


if __name__ == '__main__':
    device = 'cpu'

    target_angle = 0 # np.pi/4 -- increase torque for this angle
    agent = Agent().to(device)
    env = make_env(target_angle=target_angle, train=True)

    # Tracking for primary metrics
    train_returns = []
    train_lengths = []
    action_means = []
    action_stds = []
    cumulative_steps = [0]
    per_episode_throughput = []
    train_start_time = time.time()

    for episode in range(50):
        agent.train()
        t0 = time.time()
        episode_return, ep_steps, a_mean, a_std = run_episode(env, agent)
        dt = max(time.time() - t0, 1e-9)
        train_returns.append(episode_return)
        train_lengths.append(ep_steps)
        action_means.append(a_mean)
        action_stds.append(a_std)
        cumulative_steps.append(cumulative_steps[-1] + ep_steps)
        per_episode_throughput.append(ep_steps / dt)
        print(f"train episode={episode:02} return={episode_return:.1f} length={ep_steps} angle={env.get_wrapper_attr('state')[0]:.2f}")
 
    print('Memory buffer has retained', min(agent.memory.entries, agent.memory.max_size), 'out of', agent.memory.entries, 'experiences')

    env = make_env(target_angle=target_angle, train=False)
    # Wrap env to record only the first evaluation episode
    env = RecordVideo(env, video_folder=".", name_prefix="pendulum_episode",
                      episode_trigger=lambda e: e == 0)

    for episode in range(50):
        with torch.inference_mode():
            agent.eval()
            episode_return, _, _, _ = run_episode(env, agent, record=episode == 0)
            print(f"test  episode={episode}  return={episode_return:.1f}")

    # Compute primary metrics
    total_steps = cumulative_steps[-1]
    train_time = max(time.time() - train_start_time, 1e-9)
    throughput = total_steps / train_time
    threshold = float(os.getenv("TARGET_RETURN", "-350"))
    baseline = -1600.0
    # Moving average for stability (window=10)
    def moving_avg(x, w=10):
        if len(x) == 0:
            return []
        w = max(1, min(w, len(x)))
        c = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0.0))
        ma = (c[w:] - c[:-w]) / w
        # pad to original length
        pad = [ma[0]] * (w - 1)
        return np.concatenate([pad, ma])

    returns_ma = moving_avg(train_returns, w=10)
    # Steps-to-threshold based on moving average
    steps_to_threshold = None
    for i, r in enumerate(returns_ma):
        if r >= threshold:
            steps_to_threshold = cumulative_steps[i+1]
            break
    # AUC for returns vs cumulative steps (episode endpoints)
    x_steps = np.array(cumulative_steps[1:], dtype=float)
    y_returns = np.array(train_returns, dtype=float)
    # Normalized AUC in [0, 1] based on baseline and threshold
    denom = max(1e-9, (threshold - baseline))
    norm_scores = np.clip((y_returns - baseline) / denom, 0.0, 1.0) if len(y_returns) else np.array([])
    auc_norm = float(np.trapz(norm_scores, x_steps)) if len(x_steps) > 1 else 0.0
    auc_norm_frac = auc_norm / x_steps[-1] if len(x_steps) > 0 else 0.0

    # Create a richer progress figure (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # 1) Returns
    ax = axes[0, 0]
    ax.plot(range(len(train_returns)), train_returns, label="Return per episode", color="#1f77b4")
    ax.plot(range(len(returns_ma)), returns_ma, label="Return (10-ep MA)", color="#ff7f0e")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
    ax.axhline(baseline, color="gray", linestyle=":", label=f"Baseline {baseline}")
    ax.set_title("Training Return per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    if steps_to_threshold is not None:
        ax.annotate(f"Steps-to-threshold ≈ {int(steps_to_threshold)}",
                    xy=(len(train_returns)-1, returns_ma[-1]), xycoords='data',
                    xytext=(0.5, 0.1), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9)
    ax.legend()

    # 2) Episode length
    ax = axes[0, 1]
    ax.plot(range(len(train_lengths)), train_lengths, label="Episode length", color="#2ca02c")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    # 3) Critic loss (MA)
    ax = axes[0, 2]
    critic_ma = moving_avg(agent.critic_losses, w=50)
    ax.plot(range(len(critic_ma)), critic_ma, label="Critic loss (MA)", color="#d62728")
    ax.set_title("Critic TD Error Trend")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.legend()

    # 4) Alpha and alpha loss
    ax = axes[1, 0]
    ax.plot(range(len(agent.alphas)), agent.alphas, label="Alpha (temperature)", color="#9467bd")
    ax.set_title("Entropy Temperature α")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Alpha")
    ax2 = ax.twinx()
    alpha_loss_ma = moving_avg(agent.alpha_losses, w=50)
    ax2.plot(range(len(alpha_loss_ma)), alpha_loss_ma, label="Alpha loss (MA)", color="#8c564b")
    ax2.set_ylabel("Alpha loss")
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 5) Q1–Q2 gap (MA)
    ax = axes[1, 1]
    gap_ma = moving_avg(agent.q_gaps, w=50)
    ax.plot(range(len(gap_ma)), gap_ma, label="|Q1 - Q2| (MA)", color="#e377c2")
    ax.set_title("Q-function Disagreement")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Mean absolute gap")
    ax.legend()

    # 6) Action stats
    ax = axes[1, 2]
    ax.plot(range(len(action_means)), action_means, label="Action mean", color="#17becf")
    ax.plot(range(len(action_stds)), action_stds, label="Action std", color="#bcbd22")
    ax.set_title("Action Statistics per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.legend()

    # Global figure note and spacing
    title = (
        f"Training Diagnostics | total steps={total_steps}, "
        f"throughput={throughput:.1f} steps/s, norm AUC={auc_norm_frac:.3f} (baseline={baseline}, threshold={threshold})"
    )
    fig.suptitle(title, fontsize=12)
    # Make space so titles/labels don't overlap
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig("progress.pdf", bbox_inches="tight")

    plt.matshow(agent.memory.sars.detach().cpu().numpy().T, cmap="viridis", aspect="auto")
    plt.yticks(range(8), ["cos(theta)", "sin(theta)", "theta_dot", "torque action", "reward", "cos(theta')", "sin(theta')", "theta_dot'"])
    plt.xlabel("Experience index (time)")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig("buffer.png", bbox_inches="tight", dpi=300)
    print("Saved progress plots to progress.pdf and ordered experience buffer to buffer.png")
