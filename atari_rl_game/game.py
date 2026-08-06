from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    width: int = 160
    height: int = 210
    obs_width: int = 84
    obs_height: int = 84
    paddle_width: int = 28
    paddle_height: int = 5
    paddle_speed: float = 6.0
    ball_size: int = 5
    ball_speed_x: float = 2.5
    ball_speed_y: float = 3.2
    max_steps: int = 5_000
    starting_lives: int = 3


@dataclass
class Entity:
    x: float
    y: float
    w: int
    h: int
    vx: float
    vy: float
    kind: str

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2


def intersects(a: Entity, b: Entity) -> bool:
    return not (
        a.right < b.left
        or a.left > b.right
        or a.bottom < b.top
        or a.top > b.bottom
    )


class NeonInterceptGame:
    """Paddle-ball Atari-style game tuned for reinforcement learning."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.rng = np.random.default_rng()
        self.stars = np.empty((0, 3), dtype=np.int16)
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, int | float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.config
        self.paddle = Entity(
            x=(cfg.width - cfg.paddle_width) / 2,
            y=cfg.height - 18,
            w=cfg.paddle_width,
            h=cfg.paddle_height,
            vx=0.0,
            vy=0.0,
            kind="paddle",
        )
        self.ball = Entity(0, 0, cfg.ball_size, cfg.ball_size, 0.0, 0.0, "ball")
        self.steps = 0
        self.score = 0
        self.lives = cfg.starting_lives
        self.hits = 0
        self.misses = 0
        self.streak = 0
        self.done = False
        self.stars = np.column_stack(
            (
                self.rng.integers(0, cfg.width, size=32),
                self.rng.integers(0, cfg.height, size=32),
                self.rng.integers(1, 3, size=32),
            )
        ).astype(np.int16)
        self._serve_ball(toward_paddle=True)
        return self.info()

    def info(self) -> dict[str, int | float]:
        return {
            "score": int(self.score),
            "lives": int(self.lives),
            "steps": int(self.steps),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "streak": int(self.streak),
            "ball_x": float(self.ball.x),
            "ball_y": float(self.ball.y),
            "paddle_x": float(self.paddle.x),
        }

    def step(self, action: int) -> tuple[float, bool, bool, dict[str, int | float]]:
        if self.done:
            return 0.0, True, False, self.info()

        cfg = self.config
        self.steps += 1
        old_target_distance = self._target_distance()

        if action == 1:
            self.paddle.vx = -cfg.paddle_speed
        elif action == 2:
            self.paddle.vx = cfg.paddle_speed
        else:
            self.paddle.vx = 0.0

        self.paddle.x = float(
            np.clip(self.paddle.x + self.paddle.vx, 2, cfg.width - self.paddle.w - 2)
        )

        self.ball.x += self.ball.vx
        self.ball.y += self.ball.vy
        reward = 0.01

        if self.ball.left <= 2:
            self.ball.x = 2
            self.ball.vx = abs(self.ball.vx)
            reward += 0.02
        elif self.ball.right >= cfg.width - 2:
            self.ball.x = cfg.width - self.ball.w - 2
            self.ball.vx = -abs(self.ball.vx)
            reward += 0.02

        if self.ball.top <= 10:
            self.ball.y = 10
            self.ball.vy = abs(self.ball.vy)
            reward += 0.05

        if self.ball.vy > 0 and intersects(self.ball, self.paddle):
            reward += self._bounce_from_paddle()

        if self.ball.top > cfg.height:
            self.lives -= 1
            self.misses += 1
            self.streak = 0
            reward -= 8.0
            if self.lives > 0:
                self._serve_ball(toward_paddle=True)

        new_target_distance = self._target_distance()
        reward += 0.15 * (old_target_distance - new_target_distance)
        reward -= 0.015 * new_target_distance

        terminated = self.lives <= 0
        truncated = self.steps >= cfg.max_steps
        self.done = terminated or truncated
        if terminated:
            reward -= 10.0
        elif truncated:
            reward += 10.0 + self.streak * 0.2

        return float(reward), bool(terminated), bool(truncated), self.info()

    def state_vector(self) -> np.ndarray:
        cfg = self.config
        paddle_center = self.paddle.center_x
        ball_center = self.ball.center_x
        target_x = self.predicted_landing_x()

        return np.array(
            [
                paddle_center / cfg.width,
                ball_center / cfg.width,
                self.ball.y / cfg.height,
                self.ball.vx / (cfg.ball_speed_x * 2.2),
                self.ball.vy / (cfg.ball_speed_y * 2.2),
                (target_x - paddle_center) / cfg.width,
                self.lives / cfg.starting_lives,
                min(self.streak, 20) / 20,
            ],
            dtype=np.float32,
        )

    def predicted_landing_x(self) -> float:
        cfg = self.config
        if self.ball.vy <= 0:
            return self.ball.center_x

        steps_until_paddle = (self.paddle.top - self.ball.bottom) / self.ball.vy
        raw_x = self.ball.center_x + self.ball.vx * max(0.0, steps_until_paddle)
        min_x = self.ball.w / 2 + 2
        max_x = cfg.width - self.ball.w / 2 - 2
        span = max_x - min_x
        if span <= 0:
            return float(np.clip(raw_x, min_x, max_x))

        reflected = (raw_x - min_x) % (2 * span)
        if reflected > span:
            reflected = 2 * span - reflected
        return float(min_x + reflected)

    def render_rgb(self) -> np.ndarray:
        cfg = self.config
        frame = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        frame[:, :] = (4, 5, 14)

        star_offset = self.steps % cfg.height
        for x, y, speed in self.stars:
            yy = (int(y) + star_offset * int(speed)) % cfg.height
            brightness = 35 + int(speed) * 34
            frame[yy : yy + 1, x : x + 1] = (brightness, brightness, brightness)

        self._draw_hud(frame)
        self._rect_xy(frame, 0, self.paddle.y + self.paddle.h + 7, cfg.width, 1, (25, 34, 54))
        self._rect(frame, self.paddle, (40, 255, 120))
        self._rect_xy(
            frame,
            self.paddle.x + 2,
            self.paddle.y - 1,
            self.paddle.w - 4,
            2,
            (160, 255, 255),
        )
        self._rect(frame, self.ball, (255, 236, 82))
        self._rect_xy(frame, self.ball.x + 1, self.ball.y + 1, 2, 2, (255, 255, 255))
        return frame

    def render_observation(self) -> np.ndarray:
        cfg = self.config
        rgb = self.render_rgb()
        y_idx = np.linspace(0, cfg.height - 1, cfg.obs_height).astype(np.int16)
        x_idx = np.linspace(0, cfg.width - 1, cfg.obs_width).astype(np.int16)
        small = rgb[y_idx][:, x_idx]
        gray = (
            0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
        ).astype(np.uint8)
        return gray[:, :, None]

    def _bounce_from_paddle(self) -> float:
        cfg = self.config
        relative_hit = (self.ball.center_x - self.paddle.center_x) / (self.paddle.w / 2)
        relative_hit = float(np.clip(relative_hit, -1.0, 1.0))
        speed_bonus = min(1.8, self.hits * 0.035)

        self.ball.y = self.paddle.top - self.ball.h - 0.1
        self.ball.vy = -(cfg.ball_speed_y + speed_bonus)
        self.ball.vx = relative_hit * (cfg.ball_speed_x + speed_bonus)
        if abs(self.ball.vx) < 0.75:
            self.ball.vx = float(self.rng.choice([-0.75, 0.75]))

        self.hits += 1
        self.streak += 1
        self.score += 10 + min(self.streak, 10)
        return 5.0 + min(self.streak, 10) * 0.3

    def _serve_ball(self, toward_paddle: bool) -> None:
        cfg = self.config
        self.ball.x = float(self.rng.integers(20, cfg.width - 20))
        self.ball.y = float(cfg.height * 0.35)
        self.ball.vx = float(self.rng.uniform(-cfg.ball_speed_x, cfg.ball_speed_x))
        if abs(self.ball.vx) < 0.9:
            self.ball.vx = float(self.rng.choice([-1.2, 1.2]))
        self.ball.vy = cfg.ball_speed_y if toward_paddle else -cfg.ball_speed_y

    def _target_distance(self) -> float:
        return abs(self.predicted_landing_x() - self.paddle.center_x) / self.config.width

    def _draw_hud(self, frame: np.ndarray) -> None:
        cfg = self.config
        self._rect_xy(frame, 0, 0, cfg.width, 8, (12, 18, 34))
        for idx in range(self.lives):
            self._rect_xy(frame, 4 + idx * 8, 2, 5, 3, (60, 255, 120))

        streak_width = int(min(self.streak, 20) / 20 * 42)
        self._rect_xy(frame, cfg.width - 48, 2, 42, 3, (45, 45, 70))
        self._rect_xy(frame, cfg.width - 48, 2, streak_width, 3, (255, 230, 90))

    def _rect(self, frame: np.ndarray, ent: Entity, color: tuple[int, int, int]) -> None:
        self._rect_xy(frame, ent.x, ent.y, ent.w, ent.h, color)

    def _rect_xy(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
        color: tuple[int, int, int],
    ) -> None:
        if w <= 0 or h <= 0:
            return
        x0 = int(max(0, round(x)))
        y0 = int(max(0, round(y)))
        x1 = int(min(self.config.width, round(x + w)))
        y1 = int(min(self.config.height, round(y + h)))
        if x0 < x1 and y0 < y1:
            frame[y0:y1, x0:x1] = color
