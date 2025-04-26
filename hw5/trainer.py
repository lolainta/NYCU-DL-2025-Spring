import ale_py
import argparse
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
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


class Trainer:
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
        logger.info(f"Environment: {self.env.spec.id}")  # type: ignore
        logger.info(f"Action Space: {self.env.action_space}")
        logger.info(f"Observation Space: {self.env.observation_space}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.agent = DQNAgent(self.env, args=args)
        self.agent.train(args)
        self.preprocessor = DummyPreprocessor()

        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.episode = 0
        self.env_step = 0
        self.best_reward = 0 if self.env == "cartpole" else -21

        self.learn_per_step = args.learn_per_step

        self.eval_episodes = args.eval_episodes

    def run(self, episodes=1000):
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Training...", total=episodes)
            eval_task = progress.add_task(
                "[cyan]Episode 0: Evaluating...", total=self.episode
            )
            for ep in range(episodes):
                progress.update(task, description=f"[cyan]Episode {ep}: Training...")
                self.episode = ep
                if self.env_step > 2e6:
                    logger.info(f"Reached 2M steps, stopping training.")
                    break
                if self.episode % 20 == 0:
                    logger.info(
                        f"Episode {self.episode}: Environment Step: {self.env_step/1000:.2f}k, epsilon: {self.epsilon:.4f}"
                    )

                self.train()

                if (ep + 1) % (episodes // 20) == 0:
                    model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                    torch.save(self.agent.q_net.state_dict(), model_path)
                    logger.info(f"Saved model checkpoint to {model_path}")

                if (ep + 1) % (episodes // 50) == 0:
                    eval_rewards = []
                    progress.reset(eval_task)
                    progress.update(
                        eval_task, description=f"[cyan]Episode {ep}: Evaluating..."
                    )
                    for _ in range(self.eval_episodes):
                        eval_rewards.append(self.evaluate())
                        progress.update(eval_task, advance=1)
                    eval_reward = sum(eval_rewards) / len(eval_rewards)
                    logger.info(
                        f"Episode {ep} - Eval Reward: {eval_reward:.2f} {eval_rewards}"
                    )

                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        model_path = os.path.join(self.save_dir, "best_model.pt")
                        torch.save(self.agent.q_net.state_dict(), model_path)
                        logger.info(
                            f"Saved new best model to {model_path} with reward {eval_reward}"
                        )
                    wandb.log(
                        {
                            "Episode": ep,
                            "Env Step Count": self.env_step,
                            "Eval Reward": eval_reward,
                        }
                    )
                progress.update(task, advance=1)

    def train(self):
        avg_loss = 0

        obs, _ = self.env.reset(seed=random.randint(0, 10000))
        state = self.preprocessor.reset(obs)

        done = False
        total_reward = 0

        while not done:
            action = self.agent.select_action(state, self.epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.preprocessor.step(next_obs)
            done = terminated or truncated

            self.agent.memory.add((state, action, reward, next_state, done), 1)
            state = next_state
            total_reward += float(reward)
            self.env_step += 1

            for _ in range(self.learn_per_step):
                loss = self.agent.learn()
                if loss != float("-inf") and self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                if loss != float("-inf"):
                    avg_loss += loss
        avg_loss /= self.learn_per_step
        wandb.log(
            {
                "Episode": self.episode,
                "Env Step Count": self.env_step,
                "Total Reward": total_reward,
                "Epsilon": self.epsilon,
                "Loss": avg_loss,
            }
        )

    def evaluate(self):
        obs, _ = self.env.reset(seed=random.randint(0, 10000))
        state = self.preprocessor.reset(obs)
        frames = [self.env.render()]

        done = False
        total_reward = 0

        while not done:
            action = self.agent.select_action(state, 0.0)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            state = self.preprocessor.step(next_obs)
            done = terminated or truncated
            total_reward += float(reward)
            frames.append(self.env.render())

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
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--update-period", type=int, default=1000)
    parser.add_argument("--learn-per-step", type=int, default=1)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of eval episodes",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=3,
        help="Number of steps for multistep rewards",
    )

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.env, args.exp)
    args.env_name = "ALE/Pong-v5" if args.env == "pong" else "CartPole-v1"
    if args.env == "pong":
        gym.register_envs(ale_py)

    wandb.init(
        project="DLP-Lab5-DQN",
        name=f"{args.exp}",
        tags=[args.env, "train"],
        save_code=True,
        settings=wandb.Settings(code_dir="."),
    )
    wandb.config.update(args)

    console = Console()
    logger.remove()
    logger.add(
        lambda msg: console.print(msg, end=""),
        level=args.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level.icon} | {message}",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(f"{args}")
    trainer = Trainer(args)
    trainer.run(episodes=500)
    trainer.evaluate()
