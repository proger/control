import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.envs.classic_control import PendulumEnv
from gymnasium.wrappers import RescaleAction, TimeLimit, RecordVideo


class NoisyPendulum(PendulumEnv):
    def __init__(self, target_angle: float = 0, g: float = 10.0, eps: float = 0.0, *args, **kwargs):
        super().__init__(g=g, *args, **kwargs)
        self.eps = eps
        self.target_angle = self.angle_normalize(target_angle)
        self.max_speed = 8
        self.max_torque = 2.0  # you want more torque for weird angles

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(self, th, thdot, u):
        cost = np.abs(self.target_angle - self.angle_normalize(th)) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
        return -cost

    def step(self, u):
        u = u * self.max_torque
        th, thdot = self.state
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
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
    env = TimeLimit(
        RescaleAction(
            NoisyPendulum(target_angle=target_angle, render_mode="rgb_array", g=g, eps=eps),
            min_action=-1,
            max_action=1,
        ),
        max_episode_steps=200,
    )
    return env


def xavier(shape, scale=1.0):
    fan_in = shape[0]
    limit = scale * math.sqrt(6.0 / fan_in)
    return mx.random.uniform(low=-limit, high=limit, shape=shape)


def init_mlp(sizes):
    params = {}
    for i in range(len(sizes) - 1):
        params[f"l{i+1}"] = {
            "w": xavier((sizes[i], sizes[i + 1])),
            "b": mx.zeros((sizes[i + 1],)),
        }
    return params


def linear(p, x):
    return x @ p["w"] + p["b"]


def mlp_forward(params, x, activation=True):
    n = len(params)
    for i in range(1, n):
        x = nn.relu(linear(params[f"l{i}"], x))
    x = linear(params[f"l{n}"], x)
    if activation:
        return x
    return x


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
    def __init__(self, min_size, max_size, state_dim=3, action_dim=1):
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
        row = np.concatenate([s, a, [r], s2]).astype(np.float32)
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
    def __init__(self):
        self.state_dim = 3
        self.action_dim = 1
        self.batch_size = 200

        self.actor = init_mlp([self.state_dim, 256, 256, self.action_dim * 2])
        self.q1 = init_mlp([self.state_dim + self.action_dim, 256, 256, 1])
        self.q2 = init_mlp([self.state_dim + self.action_dim, 256, 256, 1])
        # Targets (deep clone of pytree of arrays)
        def tree_clone(tree):
            if isinstance(tree, dict):
                return {k: tree_clone(v) for k, v in tree.items()}
            # create a new array with same values
            return tree + mx.zeros_like(tree)
        self.q1_target = tree_clone(self.q1)
        self.q2_target = tree_clone(self.q2)

        self.alpha_log = mx.array(math.log(1.0), dtype=mx.float32)
        # Separate optimizers to match MLX API expectations
        self.opt_q = optim.Adam(learning_rate=1e-3)
        self.opt_actor = optim.Adam(learning_rate=1e-3)
        self.opt_alpha = optim.Adam(learning_rate=6e-4)

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
        self.training = True

    # Mode toggles for parity
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def actor_forward(self, s, params=None):
        p = self.actor if params is None else params
        x = s
        x = nn.relu(linear(p["l1"], x))
        x = nn.relu(linear(p["l2"], x))
        x = linear(p["l3"], x)
        return x

    def q_forward(self, q, s, a):
        x = mx.concatenate([s, a], axis=-1)
        x = nn.relu(linear(q["l1"], x))
        x = nn.relu(linear(q["l2"], x))
        x = linear(q["l3"], x)
        return x

    # Match torch Agent.forward behavior
    def __call__(self, state):
        if self.learning_steps == 0:
            return (np.random.rand(1).astype(np.float32) * 2 - 1)
        x = mx.array(state.reshape(1, -1))
        mean_logstd = self.actor_forward(x)
        deterministic = self.training
        a, _ = reparametrize(mean_logstd, deterministic=deterministic)
        return np.asarray(a)[0]

    def learn(self, s, a, r, s2):
        if not isinstance(s, np.ndarray):
            s = np.asarray(s, dtype=np.float32)
        if not isinstance(a, np.ndarray):
            a = np.asarray(a, dtype=np.float32)
        if not isinstance(s2, np.ndarray):
            s2 = np.asarray(s2, dtype=np.float32)
        self.memory.put(s, a, float(r), s2)

        if not self.memory.has_enough():
            return

        self.learning_steps += 1
        s_b, a_b, r_b, s2_b = self.memory.sample(self.batch_size)
        self.update_critic(s_b, a_b, r_b, s2_b)
        if self.learning_steps % 3 == 0:
            for _ in range(3):
                self.update_actor(s_b)
            self.ema_update(self.q1, self.q1_target, self.ewma_forget)
            self.ema_update(self.q2, self.q2_target, self.ewma_forget)

    def update_critic(self, s_b, a_b, r_b, s2_b):
        td = self.td_discount
        alpha_log = self.alpha_log

        def loss_q(q1, q2):
            mean_logstd_s2 = self.actor_forward(s2_b)
            a2, logp2 = reparametrize(mean_logstd_s2, deterministic=False)
            q1_t = self.q_forward(self.q1_target, s2_b, a2)
            q2_t = self.q_forward(self.q2_target, s2_b, a2)
            q_t_min = mx.minimum(q1_t, q2_t)
            v_next = q_t_min - mx.exp(alpha_log) * logp2
            y = r_b + td * v_next
            q1_pred = self.q_forward(q1, s_b, a_b)
            q2_pred = self.q_forward(q2, s_b, a_b)
            return mx.mean((q1_pred - y) ** 2 + (q2_pred - y) ** 2)

        g_q1, g_q2 = mx.grad(loss_q, argnums=(0, 1))(self.q1, self.q2)
        # For metrics, compute preds before update
        q1_pred = self.q_forward(self.q1, s_b, a_b)
        q2_pred = self.q_forward(self.q2, s_b, a_b)
        loss_val = loss_q(self.q1, self.q2)
        params = {"q1": self.q1, "q2": self.q2}
        grads = {"q1": g_q1, "q2": g_q2}
        updated = self.opt_q.apply_gradients(grads, params)
        self.q1, self.q2 = updated["q1"], updated["q2"]
        mx.eval(loss_val, q1_pred, q2_pred)
        self.critic_losses.append(float(loss_val))
        self.q_gaps.append(float(mx.mean(mx.abs(q1_pred - q2_pred))))

    def update_actor(self, s_b):
        # actor loss
        def a_loss(actor_params):
            mean_logstd = self.actor_forward(s_b, params=actor_params)
            a_s, logp = reparametrize(mean_logstd, deterministic=False)
            q1_s = mx.stop_gradient(self.q_forward(self.q1, s_b, a_s))
            q2_s = mx.stop_gradient(self.q_forward(self.q2, s_b, a_s))
            qmin = mx.minimum(q1_s, q2_s)
            return mx.mean(mx.stop_gradient(mx.exp(self.alpha_log)) * logp - qmin)

        g_actor = mx.grad(a_loss)(self.actor)
        params = {"actor": self.actor}
        grads = {"actor": g_actor}
        updated = self.opt_actor.apply_gradients(grads, params)
        self.actor = updated["actor"]

        # alpha loss
        def al_loss(alpha_log_param):
            mean_logstd = self.actor_forward(s_b)
            _, logp = reparametrize(mean_logstd, deterministic=False)
            target_entropy = -1.0
            alpha = mx.exp(alpha_log_param)
            return -mx.mean(alpha * mx.stop_gradient(logp + target_entropy))

        g_alpha = mx.grad(al_loss)(self.alpha_log)
        params = {"alpha_log": self.alpha_log}
        grads = {"alpha_log": g_alpha}
        updated = self.opt_alpha.apply_gradients(grads, params)
        self.alpha_log = updated["alpha_log"]
        # clamp
        self.alpha_log = mx.clip(self.alpha_log, mx.array(math.log(0.01)), mx.array(math.log(1.0)))

        # metrics
        a_loss_val = a_loss(self.actor)
        al_loss_val = al_loss(self.alpha_log)
        mx.eval(a_loss_val, al_loss_val)
        self.actor_losses.append(float(a_loss_val))
        self.alpha_losses.append(float(al_loss_val))
        self.alphas.append(float(mx.exp(self.alpha_log)))

    @staticmethod
    def ema_update(src, tgt, tau):
        for k in src:
            if isinstance(src[k], dict):
                Agent.ema_update(src[k], tgt[k], tau)
            else:
                tgt[k] = tgt[k] * (1.0 - tau) + src[k] * tau


def run_episode(env, agent: Agent, record=False):
    state, _ = env.reset()
    episode_return = 0.0
    truncated = False
    steps = 0
    actions = []
    while not truncated:
        action = agent(np.asarray(state, dtype=np.float32))
        a = float(action.item()) if hasattr(action, "item") else float(np.asarray(action)[0])
        next_state, reward, _, truncated, _ = env.step(a)
        agent.learn(np.asarray(state, dtype=np.float32), np.array([a], dtype=np.float32), float(reward), np.asarray(next_state, dtype=np.float32))
        episode_return += reward
        state = next_state
        steps += 1
        actions.append(a)
    action_mean = float(np.mean(actions)) if actions else 0.0
    action_std = float(np.std(actions)) if actions else 0.0
    return float(episode_return), steps, action_mean, action_std


def main():
    # Mirror pendulum.py flow
    np.random.seed(0)
    mx.random.seed(0)

    target_angle = 0
    agent = Agent()
    agent.train()
    env = make_env(target_angle=target_angle, train=True)

    # Tracking
    train_returns = []
    train_lengths = []
    action_means = []
    action_stds = []
    cumulative_steps = [0]
    per_episode_throughput = []
    train_start_time = time.time()

    for episode in range(50):
        t0 = time.time()
        episode_return, ep_steps, a_mean, a_std = run_episode(env, agent)
        dt = max(time.time() - t0, 1e-9)
        train_returns.append(episode_return)
        train_lengths.append(ep_steps)
        action_means.append(a_mean)
        action_stds.append(a_std)
        cumulative_steps.append(cumulative_steps[-1] + ep_steps)
        per_episode_throughput.append(ep_steps / dt)
        try:
            angle = env.get_wrapper_attr('state')[0]
        except Exception:
            angle = float('nan')
        print(f"train episode={episode:02} return={episode_return:.1f} length={ep_steps} angle={angle:.2f}")

    print('Memory buffer has retained', min(agent.memory.entries, agent.memory.max_size), 'out of', agent.memory.entries, 'experiences')

    # Evaluation with video
    env_eval = make_env(target_angle=target_angle, train=False)
    env_eval = RecordVideo(env_eval, video_folder='.', name_prefix='pendulum_episode', episode_trigger=lambda e: e == 0)

    for episode in range(50):
        agent.eval()
        episode_return, _, _, _ = run_episode(env_eval, agent, record=episode == 0)
        print(f"test  episode={episode}  return={episode_return:.1f}")

    # Metrics and plots
    total_steps = cumulative_steps[-1]
    train_time = max(time.time() - train_start_time, 1e-9)
    throughput = total_steps / train_time
    threshold = float(os.getenv("TARGET_RETURN", "-350"))
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
    plt.savefig("progress.pdf", bbox_inches="tight")

    plt.matshow(agent.memory.buf.T, cmap="viridis", aspect="auto")
    plt.yticks(range(8), ["cos(theta)", "sin(theta)", "theta_dot", "torque action", "reward", "cos(theta')", "sin(theta')", "theta_dot'"])
    plt.xlabel("Experience index (time)")
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig("buffer.png", bbox_inches="tight", dpi=300)
    print("Saved progress plots to progress.pdf and ordered experience buffer to buffer.png")


if __name__ == "__main__":
    main()
