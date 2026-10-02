"""Evaluate a saved PPO policy in the idealized depth environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from uav_rl.envs.depth_avoidance import DepthAvoidanceEnv
from uav_rl.training import build_policy_kwargs


def evaluate(model_path: Path, episodes: int, seed: int) -> dict[str, float]:
    env = DepthAvoidanceEnv(domain_randomization=True)
    model = PPO.load(str(model_path), device="cpu")
    returns: list[float] = []
    successes = 0
    collisions = 0
    clearances: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                returns.append(total_reward)
                successes += int(bool(info.get("success", False)))
                collisions += int(bool(info.get("collision", False)))
                clearances.append(float(info["min_clearance_m"]))
                break
    env.close()
    return {
        "episodes": float(episodes),
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "mean_terminal_clearance_m": float(np.mean(clearances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()
    print(evaluate(args.model, args.episodes, args.seed))


if __name__ == "__main__":
    main()
