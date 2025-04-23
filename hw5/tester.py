import wandb
import argparse
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import torch
import gymnasium as gym
import random
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import os

from dqn import DQNAgent
from preproccess import AtariPreprocessor, DummyPreprocessor

console = Console()


class Tester:
    def __init__(self, args) -> None:
        self.save_dir = args.save_dir
        self.env = gym.make(args.env_name, render_mode="rgb_array")
        if args.env == "pong":
            self.env = AtariPreprocessing(
                self.env,
                frame_skip=1,
                grayscale_newaxis=True,
                screen_size=84,
                grayscale_obs=True,
                noop_max=30,
            )
            self.env = FrameStackObservation(self.env, 4)

        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device:", self.device)

        self.agent = DQNAgent(self.env, args=args)
        self.agent.q_net.load_state_dict(
            torch.load(args.model_path, map_location=self.device)
        )
        self.agent.q_net.to(self.device)
        self.agent.q_net.eval()

        self.preprocessor = DummyPreprocessor()

        self.episode = 0
        self.best_reward = 0 if self.env == "cartpole" else -21

    def run(self, episodes=1000):
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Testing...", total=episodes)
            total_reward = 0
            for ep in range(episodes):
                eval_reward = self.evaluate()
                logger.info(f"Episode {ep} - Test Reward: {eval_reward:.2f}")
                total_reward += eval_reward
                wandb.log(
                    {
                        "Episode": ep,
                        "Test Reward": eval_reward,
                    }
                )

                progress.update(task, advance=1)
            total_reward /= episodes
            logger.info(f"Average Test Reward: {total_reward}")

    def evaluate(self):
        obs, _ = self.env.reset(seed=random.randint(0, 10000))
        state = self.preprocessor.reset(obs)

        done = False
        total_reward = 0

        while not done:
            action = self.agent.select_action(state, 0.0)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="cartpole",
        choices=["cartpole", "pong"],
        help="Environment name",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="test",
        help="Experiment name",
    )
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained .pt model"
    )
    parser.add_argument(
        "--test-episodes", type=int, default=100, help="Number of test episodes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.env, args.exp)
    args.env_name = "ALE/Pong-v5" if args.env == "pong" else "CartPole-v1"

    wandb.init(
        project="DLP-Lab5-DQN",
        name=f"{args.env}-{args.exp}-test",
        save_code=True,
    )
    wandb.config.update(args)

    logger.remove()
    logger.add(
        lambda msg: console.print(msg, end=""),
        level=args.log_level,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level.icon} | {message}",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(f"{args}")
    tester = Tester(args)
    tester.run(episodes=args.test_episodes)
