import wandb
import argparse
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import os
import torch
import gymnasium as gym

from dqn import DQNAgent
from atari import AtariPreprocessor


console = Console()


class Tester:
    def __init__(self, args) -> None:
        self.save_dir = args.save_dir
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device:", self.device)

        self.agent = DQNAgent(args=args)
        self.agent.q_net.load_state_dict(
            torch.load(args.model_path, map_location=self.device)
        )
        self.agent.q_net.to(self.device)
        self.agent.q_net.eval()

        self.preprocessor = AtariPreprocessor()

        self.episode = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong

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
        obs, _ = self.env.reset()
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
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-test")
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
    args = parser.parse_args()

    wandb.init(
        project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True
    )
    wandb.config.update(args)

    logger.remove()
    logger.add(lambda msg: console.print(msg, end=""), level=args.log_level)

    tester = Tester(args)
    tester.run(episodes=args.test_episodes)
