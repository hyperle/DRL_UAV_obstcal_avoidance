"""PPO training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from uav_rl.envs.depth_avoidance import DepthAvoidanceEnv


class DepthStateExtractor(BaseFeaturesExtractor):
    """Compact CNN for depth plus an MLP branch for proprioceptive state."""

    def __init__(self, observation_space, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)
        depth_shape = observation_space.spaces["depth"].shape
        state_dim = observation_space.spaces["state"].shape[0]
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(depth_shape[0], 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            depth_features = int(
                self.depth_cnn(torch.zeros(1, *depth_shape, dtype=torch.float32)).shape[1]
            )
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(depth_features + 32, features_dim), nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        depth_features = self.depth_cnn(observations["depth"])
        state_features = self.state_mlp(observations["state"])
        return self.fusion(torch.cat([depth_features, state_features], dim=1))


def build_policy_kwargs() -> dict:
    return {
        "features_extractor_class": DepthStateExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "normalize_images": False,
        "net_arch": {"pi": [128, 64], "vf": [128, 64]},
    }


def train(
    *,
    timesteps: int,
    n_envs: int,
    output: Path,
    seed: int,
    device: str,
) -> Path:
    torch.set_num_threads(1)
    vector_env = make_vec_env(
        DepthAvoidanceEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"domain_randomization": True},
    )
    model = PPO(
        "MultiInputPolicy",
        vector_env,
        policy_kwargs=build_policy_kwargs(),
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
        seed=seed,
        device=device,
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    vector_env.close()
    return output.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo_depth_avoidance"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model_path = train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        output=args.output,
        seed=args.seed,
        device=args.device,
    )
    print(f"Saved PPO checkpoint: {model_path}")


if __name__ == "__main__":
    main()
