from __future__ import annotations

from typing import Any

import numpy as np

from .game import GameConfig, NeonInterceptGame

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    gym = None
    from . import spaces  # type: ignore[no-redef]


ACTIONS: dict[int, str] = {
    0: "noop",
    1: "left",
    2: "right",
    3: "up",
    4: "down",
    5: "fire",
    6: "left_fire",
    7: "right_fire",
}


BaseEnv = gym.Env if gym is not None else object


class AtariDodgeEnv(BaseEnv):
    """Gymnasium-compatible Atari-style environment.

    Observation modes:
    - pixels: 84x84 grayscale uint8 image, shaped like classic Atari input.
    - rgb: raw 160x210 RGB frame.
    - state: compact numeric vector for quick CPU experiments.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode: str = "pixels",
        render_mode: str | None = None,
        config: GameConfig | None = None,
    ) -> None:
        if obs_mode not in {"pixels", "rgb", "state"}:
            raise ValueError("obs_mode must be 'pixels', 'rgb', or 'state'")
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.game = NeonInterceptGame(config)
        self.action_space = spaces.Discrete(len(ACTIONS))

        cfg = self.game.config
        if obs_mode == "pixels":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(cfg.obs_height, cfg.obs_width, 1),
                dtype=np.uint8,
            )
        elif obs_mode == "rgb":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(cfg.height, cfg.width, 3),
                dtype=np.uint8,
            )
        else:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.5,
                shape=(16,),
                dtype=np.float32,
            )

        self._pygame: Any = None
        self._screen: Any = None
        self._clock: Any = None
        self._scale = 3

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        if gym is not None:
            super().reset(seed=seed)
        self.game.reset(seed=seed)
        return self._observation(), self.game.info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, int | float]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..{len(ACTIONS) - 1}")

        reward, terminated, truncated, info = self.game.step(int(action))
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        frame = self.game.render_rgb()
        if self.render_mode == "rgb_array":
            return frame
        if self.render_mode == "human":
            self._render_human(frame)
        return None

    def close(self) -> None:
        if self._pygame is not None:
            self._pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None

    def _observation(self) -> np.ndarray:
        if self.obs_mode == "pixels":
            return self.game.render_observation()
        if self.obs_mode == "rgb":
            return self.game.render_rgb()
        return self.game.state_vector()

    def _render_human(self, frame: np.ndarray) -> None:
        if self._pygame is None:
            try:
                import pygame
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "pygame is required for render_mode='human'. "
                    "Install it with: pip install pygame"
                ) from exc

            self._pygame = pygame
            pygame.init()
            cfg = self.game.config
            self._screen = pygame.display.set_mode(
                (cfg.width * self._scale, cfg.height * self._scale)
            )
            pygame.display.set_caption("Neon Intercept RL")
            self._clock = pygame.time.Clock()

        surf = self._pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surf = self._pygame.transform.scale(
            surf,
            (
                self.game.config.width * self._scale,
                self.game.config.height * self._scale,
            ),
        )
        self._screen.blit(surf, (0, 0))
        self._pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
