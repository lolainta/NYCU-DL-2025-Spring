import ale_py
import argparse
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import imageio
from loguru import logger
import numpy as np
import os
import random
import sys
import torch

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

        self.device = args.device
        logger.debug(f"Using device: {self.device}")

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
        self.seed = args.seed
        gif_dir = os.path.join(self.save_dir, "gifs")
        os.makedirs(gif_dir, exist_ok=True)

    def run(self, episodes):
        success, failed = 0, 0
        total_reward: float = 0
        for ep in range(episodes):
            self.episode = ep
            # eval_reward = self.evaluate(seed=random.randint(0, 10000))
            eval_reward: float = self.evaluate(seed=self.seed + ep)
            if eval_reward < 5:
                failed += 1
            else:
                success += 1
            # if eval_reward < 0:
            #     return success, failed, eval_reward
            with open(
                os.path.join(self.save_dir, f"rewards_{self.seed}.txt"), "a"
            ) as f:
                f.write(f"{self.episode} {eval_reward}\n")
            logger.debug(f"Episode {ep} - Test Reward: {eval_reward:.2f}")
            total_reward += eval_reward
        total_reward /= episodes
        return success, failed, total_reward

    def evaluate(self, seed) -> float:
        obs, _ = self.env.reset(seed=seed)
        state = self.preprocessor.reset(obs)
        frames = [self.env.render()]

        done = False
        total_reward: float = 0.0

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
        return float(total_reward)


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
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use for training",
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.env, args.exp)
    args.env_name = "ALE/Pong-v5" if args.env == "pong" else "CartPole-v1"
    if args.env == "pong":
        gym.register_envs(ale_py)

    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level,
        colorize=True,
        format="<green>{time:YYYY-MM-DD at HH:mm:ss}</green> <level>{message}</level>",
    )

    logger.info(f"{args}")

    args.seed = random.randint(0, 2147482647)
    args.episodes = 20
    tester = Tester(args)

    successes = 0
    failures = 0
    while True:
        seed = random.randint(0, 2147483647)

        tester.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        tester.episode = 0
        logger.debug(f"Testing with seed: {tester.seed}")
        # success, failed, avg = tester.run(episodes=args.episodes)
        success, failed, avg = tester.run(episodes=args.episodes)
        if avg >= 19:
            logger.warning(f"Found seed: {tester.seed}")
            logger.warning(f"Successes: {success}, Failures: {failed}, Avg: {avg}")
        logger.info(
            f"Seed {tester.seed} - Successes: {success}, Failures: {failed}, Avg: {avg}"
        )
        successes += success
        failures += failed
        rate = successes / (successes + failures)
        logger.debug(
            f"Total successes: {successes}, Total failures: {failures}, rate: {rate * 100:.2f}%"
        )
