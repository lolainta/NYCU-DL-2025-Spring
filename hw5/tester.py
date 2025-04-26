import ale_py
import argparse
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import imageio
from loguru import logger
import numpy as np
import os
import random
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn
import torch
import wandb

from dqn import DQNAgent
from preproccess import DummyPreprocessor


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

        self.num_actions = self.env.action_space.n  # type: ignore

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.agent = DQNAgent(self.env, args=args)
        self.agent.q_net.load_state_dict(
            torch.load(args.model_path, map_location=self.device)
        )
        self.agent.q_net.to(self.device)
        self.agent.q_net.eval()

        self.preprocessor = DummyPreprocessor()

        self.episode = 0
        self.best_reward = 0 if self.env == "cartpole" else -21

        self.visualize = args.visualize
        self.save_dir = args.save_dir
        gif_dir = os.path.join(self.save_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)

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
                self.episode = ep
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
        frames = [self.env.render()]

        done = False
        total_reward = 0

        while not done:
            action = self.agent.select_action(state, 0.0)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = self.preprocessor.step(next_obs)
            frames.append(self.env.render())

        if self.visualize:
            gif_path = os.path.join(self.save_dir, "gifs", f"test_{self.episode}.gif")
            imageio.mimsave(gif_path, frames, fps=30)  # type: ignore
            logger.info(f"Saved test episode frames to {gif_path}")
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
        "--episodes", type=int, default=50, help="Number of test episodes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the environment during testing",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.env, args.exp)
    args.env_name = "ALE/Pong-v5" if args.env == "pong" else "CartPole-v1"
    if args.env == "pong":
        gym.register_envs(ale_py)
    wandb.init(
        project="DLP-Lab5-DQN",
        name=f"{args.exp}",
        tags=[args.env, "test"],
        save_code=True,
        settings=wandb.Settings(code_dir="."),
    )
    wandb.config.update(args)

    console = Console()
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
    tester.run(episodes=args.episodes)
