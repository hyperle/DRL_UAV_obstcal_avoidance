import numpy as np
from gymnasium.utils.env_checker import check_env

from uav_rl.envs.depth_avoidance import (
    ACTION_SIZE,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    STATE_SIZE,
    DepthAvoidanceEnv,
)


def test_environment_api_and_spaces():
    env = DepthAvoidanceEnv(depth_noise_std=0.0)
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=17)
    assert observation["depth"].shape == (1, DEPTH_HEIGHT, DEPTH_WIDTH)
    assert observation["state"].shape == (STATE_SIZE,)
    assert env.observation_space.contains(observation)

    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(ACTION_SIZE, dtype=np.float32)
    )
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "min_clearance_m" in info


def test_reset_is_reproducible():
    first = DepthAvoidanceEnv(depth_noise_std=0.0)
    second = DepthAvoidanceEnv(depth_noise_std=0.0)
    first_obs, _ = first.reset(seed=42)
    second_obs, _ = second.reset(seed=42)
    np.testing.assert_array_equal(first_obs["depth"], second_obs["depth"])
    np.testing.assert_array_equal(first_obs["state"], second_obs["state"])


def test_actions_are_clipped_and_observation_stays_valid():
    env = DepthAvoidanceEnv(depth_noise_std=0.0)
    env.reset(seed=2)
    observation, reward, _, _, _ = env.step(np.full(ACTION_SIZE, 9.0, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert np.isfinite(reward)
