import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import ale_py
import os
from collections import deque
import wandb
from loguru import logger
from rich.console import Console

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    """
    Design the architecture of your deep Q network
    - Input size is the same as the state dimension; the output size is the same as the number of actions
    - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
    - Feel free to add any member variables/functions whenever needed
    """

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # An example:
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)


class PrioritizedReplayBuffer:
    """
    Prioritizing the samples in the replay memory by the Bellman error
    See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """

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
    def __init__(self, num_actions=2, args=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device:", self.device)

        self.num_actions = num_actions

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, args):
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.memory = deque(maxlen=args.memory_size)
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

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
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        logger.debug("Target network updated")
