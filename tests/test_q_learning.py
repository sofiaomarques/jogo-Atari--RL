from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from atari_rl_game import AtariDodgeEnv, QLearningAgent


class QLearningAgentTest(unittest.TestCase):
    def test_agent_updates_and_roundtrips_policy(self) -> None:
        env = AtariDodgeEnv(obs_mode="state")
        obs, _ = env.reset(seed=1)
        next_obs, reward, terminated, truncated, _ = env.step(1)

        agent = QLearningAgent(rng=np.random.default_rng(1))
        agent.learn(obs, 1, reward, next_obs, terminated or truncated)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.npz"
            agent.save(path)
            loaded = QLearningAgent.load(path, rng=np.random.default_rng(1))

        np.testing.assert_allclose(loaded.q_values(obs), agent.q_values(obs))


if __name__ == "__main__":
    unittest.main()
