import torch
import torch.optim as optim
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.rescale_action import RescaleAction
from gym.wrappers.time_limit import TimeLimit


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
        q_loss = F.mse_loss(self.q1(sa), target_q_value) + F.mse_loss(self.q2(sa), target_q_value)
        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()
        self.critic_losses.append(q_loss.item())

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
        self.alpha_losses.append(alpha_loss.item())

    def ema_update(self, input_net: nn.Module, target_net: nn.Module, ewma_forget: float):
        for param_target, param_input in zip(target_net.parameters(), input_net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - ewma_forget) + param_input.data * ewma_forget)


class NoisyPendulum(PendulumEnv):
    def __init__(self, g: float = 10.0, eps: float = 0.0, *args, **kwargs):
        super().__init__(g=g, *args, **kwargs)
        self.eps = eps

    def reset(self, *, seed=0):
        super().reset(seed=seed)
        eps = self.eps
        high = np.asarray([np.pi + eps, eps])
        low = np.asarray([np.pi - eps, -eps])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}


def make_env(g=10.0, train=True):
    eps = 0.1 if train else 0.0
    env = TimeLimit(RescaleAction(NoisyPendulum(render_mode='rgb_array', g=g, eps=eps),
                                  min_action=-1, max_action=1), max_episode_steps=200)
    return env


def run_episode(env, agent, record=False):
    state, _ = env.reset()
    state = torch.from_numpy(state).float()
    episode_return, truncated = 0.0, False

    if record:
        rec = VideoRecorder(env, "pendulum_episode.mp4")

    while not truncated:
        with torch.inference_mode():
            action = agent(state)
        next_state, reward, _, truncated, _ = env.step(action.item())
        next_state = torch.from_numpy(next_state).float()
        reward = torch.tensor([reward], dtype=torch.float)

        if record:
            rec.capture_frame()

        agent.learn(state, action, reward, next_state)

        episode_return += reward
        state = next_state

    if record:
        rec.close()

    return episode_return.item()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent().to(device)
    env = make_env(train=True)

    for episode in range(50):
        agent.train()
        episode_return = run_episode(env, agent)
        print(f"train episode={episode}, return={episode_return:.1f}")

    print('Memory buffer has retained', min(agent.memory.entries, agent.memory.max_size), 'out of', agent.memory.entries, 'experiences')

    env = make_env(train=False)

    for episode in range(50):
        with torch.inference_mode():
            agent.eval()
            episode_return = run_episode(env, agent, record=episode == 0)
            print(f"test  episode={episode}, return={episode_return:.1f}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(agent.actor_losses, label="actor loss")
    axes[0].legend()
    axes[1].plot(agent.critic_losses, label="critic loss")
    axes[1].legend()
    axes[2].plot(agent.alphas, label="alpha")
    axes[2].plot(agent.alpha_losses, label="alpha loss")
    axes[2].legend()
    plt.savefig("progress.pdf", bbox_inches="tight")

    plt.matshow(agent.memory.sars.detach().numpy().T, cmap="viridis", aspect="auto")
    plt.yticks(range(8), ["cos(theta)", "sin(theta)", "theta_dot", "torque action", "reward", "cos(theta')", "sin(theta')", "theta_dot'"])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.savefig("buffer.png", bbox_inches="tight")
    print("Saved progress plots to progress.pdf and ordered experience buffer to buffer.png")
