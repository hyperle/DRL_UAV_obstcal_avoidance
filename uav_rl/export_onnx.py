"""Export a trained PPO actor to ONNX and verify it with ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from torch import nn

from uav_rl.envs.depth_avoidance import DEPTH_HEIGHT, DEPTH_WIDTH, STATE_SIZE
from uav_rl.training import DepthStateExtractor  # ensure SB3 can unpickle the custom extractor


class DeterministicActor(nn.Module):
    """SB3 PPO policy mean, clipped to the normalized action range."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, depth: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features({"depth": depth, "state": state})
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        action_mean = self.policy.action_net(latent_pi)
        return torch.clamp(action_mean, -1.0, 1.0)


def export(model_path: Path, output_path: Path, atol: float = 1e-5) -> Path:
    import onnx
    import onnxruntime as ort

    model = PPO.load(str(model_path), device="cpu")
    actor = DeterministicActor(model.policy).eval()
    example_depth = torch.zeros(1, 1, DEPTH_HEIGHT, DEPTH_WIDTH, dtype=torch.float32)
    example_state = torch.zeros(1, STATE_SIZE, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        (example_depth, example_state),
        str(output_path),
        input_names=["depth", "state"],
        output_names=["action"],
        opset_version=17,
        dynamo=True,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(23)
    max_abs_error = 0.0
    for _ in range(16):
        depth = rng.uniform(0.0, 1.0, (1, 1, DEPTH_HEIGHT, DEPTH_WIDTH)).astype(np.float32)
        state = rng.uniform(-1.0, 1.0, (1, STATE_SIZE)).astype(np.float32)
        with torch.no_grad():
            reference = actor(torch.from_numpy(depth), torch.from_numpy(state)).cpu().numpy()
        deployed = session.run(["action"], {"depth": depth, "state": state})[0]
        max_abs_error = max(max_abs_error, float(np.max(np.abs(reference - deployed))))
        np.testing.assert_allclose(reference, deployed, rtol=1e-4, atol=atol)
    print(f"Exported {output_path}; max PyTorch/ORT abs error={max_abs_error:.3g}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="SB3 checkpoint (.zip or basename)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/depth_policy.onnx"))
    args = parser.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()
