from __future__ import annotations

import unittest

import numpy as np

from atari_rl_game import ACTIONS, AtariDodgeEnv


class AtariDodgeEnvTest(unittest.TestCase):
    def test_pixels_reset_and_step(self) -> None:
        env = AtariDodgeEnv(obs_mode="pixels")
        obs, info = env.reset(seed=123)

        self.assertEqual(obs.shape, (84, 84, 1))
        self.assertEqual(obs.dtype, np.uint8)
        self.assertEqual(info["lives"], 3)

        obs, reward, terminated, truncated, info = env.step(0)

        self.assertEqual(obs.shape, (84, 84, 1))
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["steps"], 1)

    def test_state_is_deterministic_for_same_seed(self) -> None:
        env_a = AtariDodgeEnv(obs_mode="state")
        env_b = AtariDodgeEnv(obs_mode="state")

        obs_a, _ = env_a.reset(seed=42)
        obs_b, _ = env_b.reset(seed=42)
        self.assertEqual(obs_a.shape, (8,))
        np.testing.assert_allclose(obs_a, obs_b)

        actions = [0, 1, 2, 2, 0, 1, 0]
        for action in actions:
            obs_a, rew_a, done_a, trunc_a, _ = env_a.step(action)
            obs_b, rew_b, done_b, trunc_b, _ = env_b.step(action)
            np.testing.assert_allclose(obs_a, obs_b)
            self.assertEqual(rew_a, rew_b)
            self.assertEqual(done_a, done_b)
            self.assertEqual(trunc_a, trunc_b)

    def test_action_space_matches_action_table(self) -> None:
        env = AtariDodgeEnv(obs_mode="state")
        self.assertEqual(env.action_space.n, len(ACTIONS))
        for action in ACTIONS:
            self.assertTrue(env.action_space.contains(action))


if __name__ == "__main__":
    unittest.main()
