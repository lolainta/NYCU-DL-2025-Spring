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


class Trainer:
    def __init__(self, args) -> None:
        self.save_dir = args.save_dir
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device:", self.device)

        self.agent = DQNAgent(args=args)
        self.preprocessor = AtariPreprocessor()

        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.episode = 0
        self.env_count = 0
        self.best_reward = 0  # Initilized to 0 for CartPole and to -21 for Pong

        # self.max_episode_steps = args.max_episode_steps
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step

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
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                self.train()

                if ep % (episodes // 20) == 0:
                    model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                    torch.save(self.agent.q_net.state_dict(), model_path)
                    logger.info(f"Saved model checkpoint to {model_path}")

                if ep % (episodes // 100) == 0:
                    eval_reward = self.evaluate()
                    if eval_reward > self.best_reward:
                        self.best_reward = eval_reward
                        model_path = os.path.join(self.save_dir, "best_model.pt")
                        torch.save(self.agent.q_net.state_dict(), model_path)
                        logger.info(
                            f"Saved new best model to {model_path} with reward {eval_reward}"
                        )
                    logger.info(
                        f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count}"
                    )
                    wandb.log(
                        {
                            "Episode": ep,
                            "Env Step Count": self.env_count,
                            "Eval Reward": eval_reward,
                        }
                    )
                progress.update(task, advance=1)

    def train(self):
        obs, _ = self.env.reset()
        state = self.preprocessor.reset(obs)

        done = False
        total_reward = 0

        while not done:
            action = self.agent.select_action(state, self.epsilon)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_state = self.preprocessor.step(next_obs)
            done = terminated or truncated

            self.agent.memory.append((state, action, reward, next_state, terminated))

            state = next_state
            total_reward += reward
            self.env_count += 1

        for _ in range(self.train_per_step - 1):
            self.agent.train()
        if self.episode % self.target_update_frequency == 0:
            logger.debug(
                f"Updating target network at episode {self.episode} and env count {self.env_count}"
            )
            self.agent.update_target_network()

        wandb.log(
            {
                "Episode": self.episode,
                "Env Step Count": self.env_count,
                "Total Reward": total_reward,
                "Epsilon": self.epsilon,
            }
        )

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
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=100)
    # parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=4)
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

    trainer = Trainer(args)
    trainer.run(episodes=10_000)
    trainer.evaluate()
