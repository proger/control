import math
import os
import time
from collections import deque

import mlx.core as mx

import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.classic_control import PendulumEnv
from gymnasium.wrappers import RescaleAction, TimeLimit, RecordVideo
from mlx.utils import tree_map


def _frame_to_obs(frame: np.ndarray, size: int = 16) -> np.ndarray:
    """Convert an RGB frame (H, W, 3) to a small grayscale vector.

    - Converts to grayscale
    - Downsamples to (size, size) using nearest-neighbor
    - Normalizes to [0, 1]
    - Flattens to shape (size*size,)
    """
    if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("Expected RGB frame with shape (H, W, 3)")

    # Grayscale luminance
    r, g, b = frame[..., 0].astype(np.float32), frame[..., 1].astype(np.float32), frame[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    h, w = gray.shape
    # Compute strides for simple nearest-neighbor downsampling
    sh = max(1, h // size)
    sw = max(1, w // size)
    ds = gray[::sh, ::sw]
    # If result is larger than target, crop; if smaller, pad (unlikely)
    ds = ds[:size, :size]
    if ds.shape != (size, size):
        # Pad with edge values to reach target size
        pad_h = size - ds.shape[0]
        pad_w = size - ds.shape[1]
        ds = np.pad(ds, ((0, pad_h), (0, pad_w)), mode='edge')

    # Normalize to [0, 1]
    ds = ds - ds.min() if ds.size else ds
    rng = (ds.max() - ds.min()) if ds.size else 1.0
    if rng > 1e-6:
        ds = ds / rng
    ds = ds.astype(np.float32).reshape(-1)
    # Invert (background is 0)
    ds = 1 - ds
    # Binarize
    #ds = np.where(ds > 0, 1, 0)
    return ds


class NoisyPendulum(PendulumEnv):
    def __init__(self, target_angle: float = 0, g: float = 10.0, eps: float = 0.0, video: bool = False, *args, **kwargs):
        super().__init__(g=g, *args, **kwargs)
        self.eps = eps
        self.target_angle = self.angle_normalize(target_angle)
        self.max_speed = 8
        self.max_torque = 2.0 # you want more torque for weird angles
        self.video = video
        if video:
            self.obs = deque([np.zeros(16*16)]*8, maxlen=8)  # at least two frames for velocity estimation spaced far enough

    def _get_obs(self):
        if self.video:
            frame = PendulumEnv.render(self)
            self.obs.append(_frame_to_obs(frame))
            obs = mx.array(np.concatenate(self.obs))
            return obs
        else:
            theta, thetadot = self.state
            return mx.array([theta - 2*thetadot * self.dt, theta, theta - thetadot * self.dt])

    def render(self):
        if self.video:
            frame = np.repeat(np.array(self._get_obs()).reshape(8 * 16, 16)[:, :, None], 3, axis=-1) * 255
            return frame
        else:
            return np.array(self._get_obs())[None, :, None]

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(self, th, thdot, u):
        cost = np.abs(self.target_angle - self.angle_normalize(th))**2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return mx.array(-cost)[None]

    def step(self, u):
        u = u * self.max_torque # rescale the action from -1..1 to -max_torque..max_torque
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = None if self.video else u  # rendering a direction arrow

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = mx.array(np.array([newth, newthdot]))
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


def make_env(target_angle: float = 0, g=10.0, train=True, video=False):
    eps = 0.1 if train else 0.0
    env = TimeLimit(
        RescaleAction(
            NoisyPendulum(target_angle=target_angle, render_mode="rgb_array", g=g, eps=eps, video=video),
            min_action=-1,
            max_action=1,
        ),
        max_episode_steps=200,
    )
    return env


def make_mlp(sizes):
    params = {}
    for i in range(len(sizes) - 1):
        fan_in = sizes[i]
        fan_out = sizes[i + 1]
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        params[f"l{i+1}"] = {
            "w": mx.random.uniform(low=-bound, high=bound, shape=(fan_in, fan_out)),
            "b": mx.random.uniform(low=-bound, high=bound, shape=(fan_out,)),
        }
    return params


def linear(p, x):
    return x @ p["w"] + p["b"]


def reparametrize(mean_logstd, deterministic=True, LOG_STD_MIN=-20.0, LOG_STD_MAX=2.0):
    mean, log_std = mx.split(mean_logstd, 2, axis=-1)
    log_std = mx.tanh(log_std)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = mx.exp(log_std)
    if deterministic:
        z = mean
    else:
        eps = mx.random.normal(shape=mean.shape)
        z = mean + std * eps
    action = mx.tanh(z)
    # per-dimension log-prob under Normal before squash
    logp_per_dim = -0.5 * (mx.log(2 * mx.array(math.pi)) + 2 * log_std + ((z - mean) / (std + 1e-6)) ** 2)
    # subtract tanh correction per dim, then sum
    logp_correction = mx.log(1 - action ** 2 + 1e-6)
    log_prob = mx.sum(logp_per_dim - logp_correction, axis=-1, keepdims=True)
    return action, log_prob


class ReplayBuffer:
    def __init__(self, min_size, max_size, state_dim, action_dim):
        self.min_size = min_size
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        cols = self.state_dim + self.action_dim + 1 + self.state_dim
        self.buf = np.zeros((self.max_size, cols), dtype=np.float32)
        self.entries = 0

    def put(self, s, a, r, s2):
        i = self.entries % self.max_size
        row = np.concatenate([s, a, r, s2])
        self.buf[i] = row
        self.entries += 1

    def has_enough(self):
        return self.entries >= self.min_size

    def sample(self, n):
        hi = min(self.entries, self.max_size)
        idx = np.random.randint(0, hi, size=(n,))
        batch = self.buf[idx].copy()
        s, a, r, s2 = np.split(batch, [self.state_dim, self.state_dim + self.action_dim, self.state_dim + self.action_dim + 1], axis=1)
        return (
            mx.array(s),
            mx.array(a),
            mx.array(r),
            mx.array(s2),
        )


class Agent:
    def __init__(self, video=False):
        self.obs_dim = 16 * 16 * 8 if video else 3
        self.state_dim = self.obs_dim  # encode me maybe
        self.hidden_dim = 512
        self.action_dim = 1
        self.batch_size = 200

        # self.state_encoder = make_mlp([self.state_dim, self.hidden_dim])
        # self.opt_state_encoder = optim.Adam(learning_rate=1e-3, bias_correction=True)

        self.actor = make_mlp([self.state_dim, self.hidden_dim, self.hidden_dim, self.action_dim * 2])
        self.opt_actor = optim.Adam(learning_rate=1e-3, bias_correction=True)

        self.q1 = make_mlp([self.state_dim + self.action_dim, self.hidden_dim, self.hidden_dim, 1])
        self.opt_q1 = optim.Adam(learning_rate=1e-3, bias_correction=True)
        self.q2 = make_mlp([self.state_dim + self.action_dim, self.hidden_dim, self.hidden_dim, 1])
        self.opt_q2 = optim.Adam(learning_rate=1e-3, bias_correction=True)

        def tree_clone(tree):
            if isinstance(tree, dict):
                return {k: tree_clone(v) for k, v in tree.items()}
            return tree + mx.zeros_like(tree)
        self.q1_target = tree_clone(self.q1)
        self.q2_target = tree_clone(self.q2)

        self.alpha_log = mx.array(math.log(1))
        self.opt_alpha = optim.Adam(learning_rate=6e-4, bias_correction=False)
        self.opt_alpha.init_single(self.alpha_log, self.opt_alpha.state)

        self.memory = ReplayBuffer(2000, 10000, state_dim=self.state_dim, action_dim=self.action_dim)

        self.td_discount = 0.99
        self.ewma_forget = 0.05

        # Logs
        self.alphas = []
        self.alpha_losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.q_gaps = []
        self.learning_steps = 0

    def actor_forward(self, params, s):
        p = params
        x = s
        x = nn.relu(linear(p["l1"], x))
        x = nn.relu(linear(p["l2"], x))
        x = linear(p["l3"], x)
        return x

    def q_forward(self, q, s, a, target=None):
        x = mx.concatenate([s, a], axis=-1)
        x = nn.relu(linear(q["l1"], x))
        x = nn.relu(linear(q["l2"], x))
        x = linear(q["l3"], x)
        if target is not None:
            return (x - target).square().mean(), x
        else:
            return -mx.inf, x

    def __call__(self, state, deterministic=True):
        if self.learning_steps == 0:
            return (np.random.rand(1).astype(np.float32) * 2 - 1)

        state = state[None, :]
        mean_logstd = self.actor_forward(self.actor, state)
        action, logprob = reparametrize(mean_logstd, deterministic=deterministic)
        return action[0]

    def learn(self, state, action, reward, next_state):
        self.memory.put(state, action, reward, next_state)

        if not self.memory.has_enough():
            return

        self.learning_steps += 1

        s_batch, a_batch, r_batch, s2_batch = self.memory.sample(self.batch_size)
        self.update_critic(s_batch, a_batch, r_batch, s2_batch)

        if self.learning_steps % 3 == 0:
            for _ in range(3):
                self.update_actor(s_batch)

            self.ema_update(self.q1, self.q1_target, self.ewma_forget)
            self.ema_update(self.q2, self.q2_target, self.ewma_forget)

    def update_critic(self, states, actions, rewards, next_states):
        next_actions, log_prob = reparametrize(
            self.actor_forward(self.actor, next_states),
            deterministic=False
        )

        td_target = mx.minimum(
            self.q_forward(self.q1_target, next_states, next_actions)[1],
            self.q_forward(self.q2_target, next_states, next_actions)[1]
        ) - self.alpha_log.exp() * log_prob

        target_q_value = rewards + self.td_discount * td_target

        q = mx.value_and_grad(self.q_forward)
        (q1_loss, q1_pred), q1_grads = q(self.q1, states, actions, target=target_q_value)
        (q2_loss, q2_pred), q2_grads = q(self.q2, states, actions, target=target_q_value)
        self.q1 = self.opt_q1.apply_gradients(q1_grads, self.q1)
        mx.eval(self.q1)
        self.q2 = self.opt_q2.apply_gradients(q2_grads, self.q2)
        mx.eval(self.q2)

        self.critic_losses.append((q1_loss + q2_loss).item())
        self.q_gaps.append((q1_pred - q2_pred).abs().mean().item())

    def actor_loss(self, actor_params, states):
        actions, logprob = reparametrize(
            self.actor_forward(actor_params, states),
            deterministic=False
        )

        actor_loss = (- mx.minimum(
            self.q_forward(self.q1_target, states, actions)[1],
            self.q_forward(self.q2_target, states, actions)[1]
        ) + self.alpha_log.exp() * logprob).mean()
        return actor_loss, logprob

    def alpha_loss(self, alpha_log, logprob):
        return -(- alpha_log.exp() * (logprob + 1)).mean()

    def update_actor(self, states):
        loss = mx.value_and_grad(self.actor_loss)
        (l, logprob), actor_grads = loss(self.actor, states)
        self.actor_losses.append(l.item())
        self.actor = self.opt_actor.apply_gradients(actor_grads, self.actor)
        mx.eval(self.actor)

        alpha_loss = mx.value_and_grad(self.alpha_loss)
        al, alpha_grads = alpha_loss(self.alpha_log, logprob)
        self.alpha_losses.append(al.item())

        self.alpha_log = self.opt_alpha.apply_single(alpha_grads, self.alpha_log, self.opt_alpha.state)
        self.alpha_log = mx.clip(self.alpha_log, math.log(0.01), math.log(1))

        self.alphas.append(self.alpha_log.exp().item())

    @staticmethod
    def ema_update(src, tgt, tau):
        for k in src:
            if isinstance(src[k], dict):
                Agent.ema_update(src[k], tgt[k], tau)
            else:
                tgt[k] = tgt[k] * (1.0 - tau) + src[k] * tau


def run_episode(env, agent: Agent, learning: bool = True):
    state, _ = env.reset()
    episode_return = 0.0
    truncated = False
    steps = 0
    actions = []

    while not truncated:
        action = agent(state, deterministic=learning)
        next_state, reward, _, truncated, _ = env.step(action.item())

        if learning:
            agent.learn(state, action, reward, next_state)

        episode_return += reward
        state = next_state
        steps += 1
        actions.append(action.item())

    action_mean = float(np.mean(actions)) if actions else 0.0
    action_std = float(np.std(actions)) if actions else 0.0
    return episode_return.item(), steps, action_mean, action_std


if __name__ == "__main__":
    np.random.seed(0)
    mx.random.seed(0)

    video = True
    target_angle = 0
    env = make_env(target_angle=target_angle, train=True, video=video)
    agent = Agent(video=video)

    # Tracking
    train_returns = []
    train_lengths = []
    action_means = []
    action_stds = []
    cumulative_steps = [0]
    per_episode_throughput = []
    train_start_time = time.time()

    for episode in range(2000):
        t0 = time.time()
        episode_return, ep_steps, a_mean, a_std = run_episode(env, agent)
        dt = max(time.time() - t0, 1e-9)
        train_returns.append(episode_return)
        train_lengths.append(ep_steps)
        action_means.append(a_mean)
        action_stds.append(a_std)
        cumulative_steps.append(cumulative_steps[-1] + ep_steps)
        per_episode_throughput.append(ep_steps / dt)
        angle = env.get_wrapper_attr('state')[0]
        print(f"train episode={episode:02} return={episode_return:.1f} length={ep_steps} angle={angle:.2f}")

    print('Memory buffer has retained', min(agent.memory.entries, agent.memory.max_size), 'out of', agent.memory.entries, 'experiences')

    env_eval = make_env(target_angle=target_angle, train=False, video=video)
    env_eval = RecordVideo(env_eval, video_folder='.', name_prefix='pendulum_episode', episode_trigger=lambda e: e == 0)

    for episode in range(2):
        episode_return, _, _, _ = run_episode(env_eval, agent, learning=False)
        print(f"test  episode={episode}  return={episode_return:.1f}")

    # Metrics and plots
    total_steps = cumulative_steps[-1]
    train_time = max(time.time() - train_start_time, 1e-9)
    throughput = total_steps / train_time
    threshold = -350.0
    baseline = -1600.0

    def moving_avg(x, w=10):
        if len(x) == 0:
            return []
        w = max(1, min(w, len(x)))
        c = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0.0))
        ma = (c[w:] - c[:-w]) / w
        pad = [ma[0]] * (w - 1)
        return np.concatenate([pad, ma])

    returns_ma = moving_avg(train_returns, w=10)
    steps_to_threshold = None
    for i, r in enumerate(returns_ma):
        if r >= threshold:
            steps_to_threshold = cumulative_steps[i + 1]
            break

    x_steps = np.array(cumulative_steps[1:], dtype=float)
    y_returns = np.array(train_returns, dtype=float)
    denom = max(1e-9, (threshold - baseline))
    norm_scores = np.clip((y_returns - baseline) / denom, 0.0, 1.0) if len(y_returns) else np.array([])
    auc_norm = float(np.trapz(norm_scores, x_steps)) if len(x_steps) > 1 else 0.0
    auc_norm_frac = auc_norm / x_steps[-1] if len(x_steps) > 0 else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax = axes[0, 0]
    ax.plot(range(len(train_returns)), train_returns, label="Return per episode", color="#1f77b4")
    ax.plot(range(len(returns_ma)), returns_ma, label="Return (10-ep MA)", color="#ff7f0e")
    ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
    ax.axhline(baseline, color="gray", linestyle=":", label=f"Baseline {baseline}")
    ax.set_title("Training Return per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    if steps_to_threshold is not None:
        ax.annotate(
            f"Steps-to-threshold ≈ {int(steps_to_threshold)}",
            xy=(len(train_returns) - 1, returns_ma[-1]),
            xycoords='data',
            xytext=(0.5, 0.1),
            textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9,
        )
    ax.legend()

    ax = axes[0, 1]
    ax.plot(range(len(train_lengths)), train_lengths, label="Episode length", color="#2ca02c")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()

    ax = axes[0, 2]
    critic_ma = moving_avg(agent.critic_losses, w=50)
    ax.plot(range(len(critic_ma)), critic_ma, label="Critic loss (MA)", color="#d62728")
    ax.set_title("Critic TD Error Trend")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Loss")
    ax.legend()

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

    ax = axes[1, 1]
    gap_ma = moving_avg(agent.q_gaps, w=50)
    ax.plot(range(len(gap_ma)), gap_ma, label="|Q1 - Q2| (MA)", color="#e377c2")
    ax.set_title("Q-function Disagreement")
    ax.set_xlabel("Update step")
    ax.set_ylabel("Mean absolute gap")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(range(len(action_means)), action_means, label="Action mean", color="#17becf")
    ax.plot(range(len(action_stds)), action_stds, label="Action std", color="#bcbd22")
    ax.set_title("Action Statistics per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.legend()

    title = (
        f"Training Diagnostics | total steps={total_steps}, "
        f"throughput={throughput:.1f} steps/s, norm AUC={auc_norm_frac:.3f} (baseline={baseline}, threshold={threshold})"
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig("progress_mlx.pdf", bbox_inches="tight")

    plt.matshow(agent.memory.buf.T, cmap="viridis", aspect="auto")
    if not video:
        plt.yticks(range(8), ["cos(theta)", "sin(theta)", "theta_dot", "torque action", "reward", "cos(theta')", "sin(theta')", "theta_dot'"])
    else:
        plt.yticks([])
        plt.ylabel(f"Features (2×{agent.state_dim} + action + reward)")
    plt.xlabel("Experience index (time)")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig("buffer.png", bbox_inches="tight", dpi=300)
    print("Saved progress plots to progress.pdf and ordered experience buffer to buffer.png")
