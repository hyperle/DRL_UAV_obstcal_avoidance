"""Fast, idealized 3-D depth-image environment for initial policy research.

This renderer projects spherical obstacles into a low-resolution pinhole-camera
depth image. It is deliberately fast and deterministic given a seed; it is not
a substitute for a calibrated stereo renderer or a flight-dynamics simulator.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DEPTH_HEIGHT = 32
DEPTH_WIDTH = 48
STATE_SIZE = 11
ACTION_SIZE = 4

# These are initial training-envelope values, not verified airframe limits.
MAX_FORWARD_MPS = 1.5
MAX_LATERAL_MPS = 0.8
MAX_VERTICAL_MPS = 0.6
MAX_YAW_RATE_RADPS = 1.2
MAX_GOAL_RANGE_M = 10.0
MAX_DEPTH_M = 8.0
MIN_DEPTH_M = 0.15


class DepthAvoidanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous velocity-control task using a depth image and compact state.

    Observation dictionary:
      * ``depth``: normalized depth, shape ``(1, 32, 48)``, float32 in [0, 1].
        1.0 means no return / beyond max range.
      * ``state``: goal vector in body coordinates, body velocity, yaw rate and
        previous normalized action, all normalized to approximately [-1, 1].

    Action: normalized body-frame ``[forward, lateral, vertical, yaw_rate]``.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        max_steps: int = 300,
        dt: float = 0.15,
        obstacle_count: int = 12,
        depth_noise_std: float = 0.005,
        domain_randomization: bool = True,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if max_steps <= 0 or dt <= 0 or obstacle_count < 0:
            raise ValueError("max_steps and dt must be positive; obstacle_count non-negative")

        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.obstacle_count = int(obstacle_count)
        self.depth_noise_std = float(depth_noise_std)
        self.domain_randomization = bool(domain_randomization)

        self.observation_space = spaces.Dict(
            {
                "depth": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1, DEPTH_HEIGHT, DEPTH_WIDTH),
                    dtype=np.float32,
                ),
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(STATE_SIZE,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32
        )

        self.position = np.zeros(3, dtype=np.float32)
        self.velocity_world = np.zeros(3, dtype=np.float32)
        self.goal = np.array([8.0, 0.0, 1.2], dtype=np.float32)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.previous_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.obstacles = np.zeros((0, 4), dtype=np.float32)  # x, y, z, radius
        self.steps = 0
        self._episode_count = 0
        self._velocity_time_constant = 0.25
        self._speed_scales = np.array(
            [MAX_FORWARD_MPS, MAX_LATERAL_MPS, MAX_VERTICAL_MPS], dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._episode_count += 1
        self.position = np.array([0.0, 0.0, 1.2], dtype=np.float32)
        self.velocity_world = np.zeros(3, dtype=np.float32)
        goal_x = float(self.np_random.uniform(7.0, 10.0))
        goal_y = float(self.np_random.uniform(-1.0, 1.0))
        goal_z = float(self.np_random.uniform(0.8, 1.8))
        self.goal = np.array([goal_x, goal_y, goal_z], dtype=np.float32)
        self.yaw = float(self.np_random.uniform(-0.12, 0.12))
        self.yaw_rate = 0.0
        self.previous_action.fill(0.0)
        self.steps = 0
        self._previous_goal_distance = float(np.linalg.norm(self.goal - self.position))

        if self.domain_randomization:
            self._velocity_time_constant = float(self.np_random.uniform(0.18, 0.36))
            scale = self.np_random.uniform(0.9, 1.1, size=3).astype(np.float32)
            self._speed_scales = np.array(
                [MAX_FORWARD_MPS, MAX_LATERAL_MPS, MAX_VERTICAL_MPS], dtype=np.float32
            ) * scale
        else:
            self._velocity_time_constant = 0.25
            self._speed_scales = np.array(
                [MAX_FORWARD_MPS, MAX_LATERAL_MPS, MAX_VERTICAL_MPS], dtype=np.float32
            )

        self.obstacles = self._sample_obstacles()
        observation = self._observation()
        return observation, self._info(min_clearance=self._min_clearance())

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_SIZE)
        action = np.clip(action, -1.0, 1.0)
        self.steps += 1

        desired_body_velocity = action[:3] * self._speed_scales
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        desired_world_velocity = np.array(
            [
                c * desired_body_velocity[0] - s * desired_body_velocity[1],
                s * desired_body_velocity[0] + c * desired_body_velocity[1],
                desired_body_velocity[2],
            ],
            dtype=np.float32,
        )
        response = 1.0 - np.exp(-self.dt / self._velocity_time_constant)
        self.velocity_world += response * (desired_world_velocity - self.velocity_world)
        self.yaw_rate = float(action[3] * MAX_YAW_RATE_RADPS)
        self.yaw = float((self.yaw + self.yaw_rate * self.dt + np.pi) % (2.0 * np.pi) - np.pi)
        self.position += self.velocity_world * self.dt

        goal_distance = float(np.linalg.norm(self.goal - self.position))
        min_clearance = self._min_clearance()
        collision = min_clearance <= 0.0
        success = goal_distance <= 0.35
        out_of_bounds = bool(
            self.position[0] < -1.0
            or self.position[0] > float(self.goal[0] + 2.0)
            or abs(float(self.position[1])) > 5.0
            or not 0.2 <= float(self.position[2]) <= 3.0
        )

        previous_distance = getattr(self, "_previous_goal_distance", goal_distance)
        reward = 2.0 * (previous_distance - goal_distance)
        reward -= 0.015 * float(np.square(action).sum())
        reward -= 0.025 * float(np.square(action - self.previous_action).sum())
        reward -= 0.01  # small time cost

        safe_clearance = 0.7
        if min_clearance < safe_clearance:
            normalized_risk = (safe_clearance - min_clearance) / safe_clearance
            reward -= 0.25 * float(normalized_risk * normalized_risk)
        if success:
            reward += 25.0
        if collision:
            reward -= 25.0
        if out_of_bounds:
            reward -= 15.0

        self.previous_action = action.copy()
        self._previous_goal_distance = goal_distance
        terminated = bool(success or collision or out_of_bounds)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        observation = self._observation()
        info = self._info(
            min_clearance=min_clearance,
            goal_distance=goal_distance,
            success=success,
            collision=collision,
            out_of_bounds=out_of_bounds,
        )
        return observation, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"step={self.steps} position={self.position.round(2).tolist()} "
            f"goal={self.goal.round(2).tolist()} "
            f"nearest_clearance={self._min_clearance():.2f}m"
        )

    def _sample_obstacles(self) -> np.ndarray:
        obstacles: list[list[float]] = []
        for _ in range(self.obstacle_count):
            for _attempt in range(100):
                x = float(self.np_random.uniform(1.2, max(1.3, float(self.goal[0] - 0.8))))
                y = float(self.np_random.uniform(-2.2, 2.2))
                z = float(self.np_random.uniform(0.45, 2.4))
                radius = float(self.np_random.uniform(0.18, 0.42))
                candidate = np.array([x, y, z], dtype=np.float32)
                if np.linalg.norm(candidate - self.position) > radius + 0.7 and np.linalg.norm(
                    candidate - self.goal
                ) > radius + 0.65:
                    obstacles.append([x, y, z, radius])
                    break
        if not obstacles:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(obstacles, dtype=np.float32)

    def _body_relative(self, world_vector: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array(
            [
                c * world_vector[0] + s * world_vector[1],
                -s * world_vector[0] + c * world_vector[1],
                world_vector[2],
            ],
            dtype=np.float32,
        )

    def _body_velocity(self) -> np.ndarray:
        return self._body_relative(self.velocity_world)

    def _get_depth(self) -> np.ndarray:
        """Project sphere silhouettes into a normalized pinhole depth image."""
        depth_m = np.full((DEPTH_HEIGHT, DEPTH_WIDTH), MAX_DEPTH_M, dtype=np.float32)
        h_fov = np.deg2rad(90.0)
        v_fov = np.deg2rad(60.0)
        fx = DEPTH_WIDTH / (2.0 * np.tan(h_fov / 2.0))
        fy = DEPTH_HEIGHT / (2.0 * np.tan(v_fov / 2.0))

        for obstacle in self.obstacles:
            relative = self._body_relative(obstacle[:3] - self.position)
            forward, lateral, vertical = (float(v) for v in relative)
            if forward <= -float(obstacle[3]):
                continue
            center_distance = float(np.linalg.norm(relative))
            radius = float(obstacle[3])
            surface_range = max(MIN_DEPTH_M, center_distance - radius)
            if surface_range >= MAX_DEPTH_M:
                continue

            horizontal_angle = float(np.arctan2(lateral, forward))
            vertical_angle = float(np.arctan2(vertical, max(1e-4, np.hypot(forward, lateral))))
            if abs(horizontal_angle) > h_fov / 2.0 + 0.5 or abs(vertical_angle) > v_fov / 2.0 + 0.5:
                continue

            center_u = (0.5 - horizontal_angle / h_fov) * DEPTH_WIDTH
            center_v = (0.5 - vertical_angle / v_fov) * DEPTH_HEIGHT
            projected_radius = radius * fx / max(0.2, forward - radius)
            projected_radius_v = radius * fy / max(0.2, forward - radius)
            u0 = max(0, int(np.floor(center_u - projected_radius)))
            u1 = min(DEPTH_WIDTH, int(np.ceil(center_u + projected_radius)) + 1)
            v0 = max(0, int(np.floor(center_v - projected_radius_v)))
            v1 = min(DEPTH_HEIGHT, int(np.ceil(center_v + projected_radius_v)) + 1)
            if u0 < u1 and v0 < v1:
                depth_m[v0:v1, u0:u1] = np.minimum(depth_m[v0:v1, u0:u1], surface_range)

        if self.depth_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.depth_noise_std, depth_m.shape)
            depth_m = np.clip(depth_m + noise.astype(np.float32), MIN_DEPTH_M, MAX_DEPTH_M)
        return np.clip(depth_m / MAX_DEPTH_M, 0.0, 1.0).astype(np.float32)[None, ...]

    def _observation(self) -> dict[str, np.ndarray]:
        goal_body = self._body_relative(self.goal - self.position)
        body_velocity = self._body_velocity()
        state = np.concatenate(
            [
                np.clip(goal_body / MAX_GOAL_RANGE_M, -1.0, 1.0),
                np.clip(
                    body_velocity
                    / np.array(
                        [MAX_FORWARD_MPS, MAX_LATERAL_MPS, MAX_VERTICAL_MPS],
                        dtype=np.float32,
                    ),
                    -1.0,
                    1.0,
                ),
                np.array([np.clip(self.yaw_rate / MAX_YAW_RATE_RADPS, -1.0, 1.0)], dtype=np.float32),
                self.previous_action,
            ]
        ).astype(np.float32)
        return {"depth": self._get_depth(), "state": state}

    def _min_clearance(self) -> float:
        if len(self.obstacles) == 0:
            return MAX_DEPTH_M
        distances = np.linalg.norm(self.obstacles[:, :3] - self.position[None, :], axis=1)
        return float(np.min(distances - self.obstacles[:, 3]))

    def _info(self, *, min_clearance: float, **extra: Any) -> dict[str, Any]:
        return {
            "step": self.steps,
            "min_clearance_m": float(min_clearance),
            **extra,
        }
