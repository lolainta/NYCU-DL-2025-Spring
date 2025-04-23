import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from loguru import logger

from models import CartPoleDQN, PongDQN


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########

        ########## END OF YOUR CODE (for Task 3) ##########
        return


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
        self.memory = deque(maxlen=args.memory_size)
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

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

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, self.batch_size)
        )

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
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_count += 1
        if self.learn_count % self.update_period == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            logger.debug(
                f"Target network updated at learn_count={self.learn_count/1000:.2f}k"
            )
