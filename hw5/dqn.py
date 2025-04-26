from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from loguru import logger

from models import CartPoleDQN, PongDQN


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, n_steps=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.n_steps = n_steps
        self.gamma = gamma

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer = deque(maxlen=n_steps)
        self.pos = 0

    def add(self, transition, error):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_steps:
            return

        # Compute multistep reward and next state
        reward, next_state, done = self._get_multistep_transition()
        state, action = self.n_step_buffer[0][:2]
        multistep_transition = (state, action, reward, next_state, done)

        priority = (abs(error) + 1e-6) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(multistep_transition)
        else:
            self.buffer[self.pos] = multistep_transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def _get_multistep_transition(self):
        reward, next_state, done = 0, None, False
        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma**idx) * r
            next_state, done = ns, d
            if done:
                break
        return reward, next_state, done

    def sample(self, batch_size):
        assert len(self.buffer) >= batch_size, "Not enough samples to sample from"

        valid_size = len(self.buffer)
        probs = self.priorities[:valid_size]
        probs = probs / probs.sum()

        indices = np.random.choice(valid_size, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = valid_size
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        match env.spec.id:
            case "CartPole-v1":
                self.input_state = 4
                self.num_actions = 2
                self.DQN = CartPoleDQN
            case "ALE/Pong-v5":
                self.input_state = 4
                self.num_actions = 6
                self.DQN = PongDQN
            case _:
                raise ValueError(f"Unsupported environment: {env}")

        self.q_net = self.DQN(self.input_state, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, args):
        self.batch_size = args.batch_size
        self.update_period = args.update_period
        self.gamma = args.discount_factor
        self.memory = PrioritizedReplayBuffer(
            args.memory_size, args.per_alpha, args.per_beta
        )

        self.target_net = self.DQN(self.input_state, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.learn_count = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = (
            torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        )
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        self.q_net.train()
        return q_values.argmax().item()

    def learn(self) -> float:
        if len(self.memory) < self.batch_size:
            return float("-inf")

        batch, indices, weights = self.memory.sample(self.batch_size)
        indices = torch.tensor(indices, dtype=torch.int64).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(
            self.device
        )
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            # target_q_values = rewards + self.gamma * next_q_values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors = target_q_values - q_values
        loss = (td_errors**2 * weights).mean()
        # loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        # self.memory.beta = min(1.0, self.memory.beta + 0.000001)

        self.learn_count += 1
        if self.learn_count % self.update_period == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            logger.debug(
                f"Target network updated at learn_count={self.learn_count/1000:.2f}k"
            )
            logger.debug(
                f"Memory size: {len(self.memory)}, beta: {self.memory.beta:.2f}"
            )
        return loss.item()
