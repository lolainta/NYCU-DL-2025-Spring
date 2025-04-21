# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

gym.register_envs(ale_py)

console = Console()
logger.remove()
logger.add(lambda msg: console.print(msg, end=""))


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
        ########## YOUR CODE HERE (5~10 lines) ##########

        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
    Preprocesing the state input of DQN for Atari
    """

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        return obs
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        return obs
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)], maxlen=self.frame_stack
        )
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        return obs
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


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
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device:", self.device)

        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.episode = 0
        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = deque(maxlen=args.memory_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        state_tensor = (
            torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        )
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        self.q_net.train()
        return q_values.argmax().item()

    def run(self, episodes=1000):
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Training...", total=episodes)
            for ep in range(episodes):
                self.episode = ep
                obs, _ = self.env.reset()

                state = self.preprocessor.reset(obs)
                done = False
                total_reward = 0
                step_count = 0

                while not done:
                    if step_count >= self.max_episode_steps:
                        logger.warning(
                            f"Episode {ep} reached max steps {self.max_episode_steps}"
                        )
                    action = self.select_action(state)
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    next_state = self.preprocessor.step(next_obs)
                    self.memory.append((state, action, reward, next_state, done))

                    for _ in range(self.train_per_step):
                        self.train()

                    state = next_state
                    total_reward += reward
                    self.env_count += 1
                    step_count += 1

                wandb.log(
                    {
                        "Episode": ep,
                        "Total Reward": total_reward,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    }
                )
                ########## YOUR CODE HERE  ##########
                # Add additional wandb logs for debugging if needed

                ########## END OF YOUR CODE ##########
                if ep % 1000 == 0:
                    model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    logger.info(f"Saved model checkpoint to {model_path}")

                if ep % 20 == 0:
                    eval_reward = self.evaluate()
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        model_path = os.path.join(self.save_dir, "best_model.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        logger.info(
                            f"Saved new best model to {model_path} with reward {eval_reward}"
                        )
                    logger.info(
                        f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}"
                    )
                    wandb.log(
                        {
                            "Episode": ep,
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                            "Eval Reward": eval_reward,
                        }
                    )
                progress.update(task, advance=1)

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = (
                torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            )
            self.q_net.eval()
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            self.q_net.train()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward

    def train(self):

        if len(self.memory) < self.replay_start_size:
            return
        if len(self.memory) < self.batch_size:
            return

        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer

        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, self.batch_size)
        )

        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(
            self.device
        )
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        wandb.log(
            {
                "Episode": self.episode,
                "Train Count": self.train_count,
                "loss": loss.item(),
                "q_mean": q_values.mean().item(),
                "q_std": q_values.std().item(),
            }
        )

        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        # if self.train_count % 1000 == 0:
        #     tqdm.write(
        #         f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}"
        #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    args = parser.parse_args()

    wandb.init(
        project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True
    )
    wandb.config.update(args)
    agent = DQNAgent(args=args)
    agent.run(episodes=10000)
