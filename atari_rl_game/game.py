from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GameConfig:
    width: int = 160
    height: int = 210
    obs_width: int = 84
    obs_height: int = 84
    player_size: int = 8
    player_speed: float = 4.0
    bullet_speed: float = 7.0
    fire_cooldown: int = 8
    max_steps: int = 4_000
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


def intersects(a: Entity, b: Entity) -> bool:
    return not (
        a.right < b.left
        or a.left > b.right
        or a.bottom < b.top
        or a.top > b.bottom
    )


class NeonInterceptGame:
    """Small deterministic arcade game tuned for reinforcement learning."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.rng = np.random.default_rng()
        self.stars = np.empty((0, 3), dtype=np.int16)
        self.reset()

    def reset(self, seed: int | None = None) -> dict[str, int | float]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self.config
        self.player = Entity(
            x=(cfg.width - cfg.player_size) / 2,
            y=cfg.height - 28,
            w=cfg.player_size,
            h=cfg.player_size,
            vx=0,
            vy=0,
            kind="player",
        )
        self.bullets: list[Entity] = []
        self.meteors: list[Entity] = []
        self.energy: list[Entity] = []
        self.enemies: list[Entity] = []
        self.cooldown = 0
        self.steps = 0
        self.score = 0
        self.lives = cfg.starting_lives
        self.done = False
        self.stars = np.column_stack(
            (
                self.rng.integers(0, cfg.width, size=40),
                self.rng.integers(0, cfg.height, size=40),
                self.rng.integers(1, 4, size=40),
            )
        ).astype(np.int16)
        return self.info()

    def info(self) -> dict[str, int | float]:
        return {
            "score": int(self.score),
            "lives": int(self.lives),
            "steps": int(self.steps),
            "bullets": len(self.bullets),
            "meteors": len(self.meteors),
            "energy": len(self.energy),
            "enemies": len(self.enemies),
        }

    def step(self, action: int) -> tuple[float, bool, bool, dict[str, int | float]]:
        if self.done:
            return 0.0, True, False, self.info()

        cfg = self.config
        self.steps += 1
        reward = 0.01

        dx = dy = 0.0
        fire = False
        if action == 1:
            dx = -cfg.player_speed
        elif action == 2:
            dx = cfg.player_speed
        elif action == 3:
            dy = -cfg.player_speed
        elif action == 4:
            dy = cfg.player_speed
        elif action == 5:
            fire = True
        elif action == 6:
            dx = -cfg.player_speed
            fire = True
        elif action == 7:
            dx = cfg.player_speed
            fire = True

        self.player.x = float(np.clip(self.player.x + dx, 2, cfg.width - self.player.w - 2))
        self.player.y = float(
            np.clip(self.player.y + dy, cfg.height * 0.45, cfg.height - self.player.h - 6)
        )

        if self.cooldown > 0:
            self.cooldown -= 1
        if fire and self.cooldown == 0:
            self.bullets.append(
                Entity(
                    x=self.player.x + self.player.w / 2 - 1,
                    y=self.player.y - 4,
                    w=2,
                    h=5,
                    vx=0,
                    vy=-cfg.bullet_speed,
                    kind="bullet",
                )
            )
            self.cooldown = cfg.fire_cooldown

        self._spawn_wave()
        self._move_entities()
        reward += self._resolve_bullet_hits()
        reward += self._resolve_player_hits()
        reward += self._remove_offscreen()

        terminated = self.lives <= 0
        truncated = self.steps >= cfg.max_steps
        self.done = terminated or truncated
        if terminated:
            reward -= 10.0
        elif truncated:
            reward += 5.0

        return float(reward), bool(terminated), bool(truncated), self.info()

    def state_vector(self) -> np.ndarray:
        cfg = self.config

        def nearest(items: list[Entity], default_y: float) -> Entity:
            if not items:
                return Entity(0, default_y, 1, 1, 0, 0, "none")
            return min(items, key=lambda ent: abs(ent.x - self.player.x) + abs(ent.y - self.player.y))

        meteor = nearest(self.meteors, 0)
        energy = nearest(self.energy, 0)
        enemy = nearest(self.enemies, 0)
        bullet_y = max((b.y for b in self.bullets), default=cfg.height)

        return np.array(
            [
                self.player.x / cfg.width,
                self.player.y / cfg.height,
                self.cooldown / cfg.fire_cooldown,
                self.lives / cfg.starting_lives,
                len(self.meteors) / 12,
                len(self.energy) / 4,
                len(self.enemies) / 4,
                meteor.x / cfg.width,
                meteor.y / cfg.height,
                meteor.vy / 5,
                energy.x / cfg.width,
                energy.y / cfg.height,
                enemy.x / cfg.width,
                enemy.y / cfg.height,
                enemy.vx / 3,
                bullet_y / cfg.height,
            ],
            dtype=np.float32,
        )

    def render_rgb(self) -> np.ndarray:
        cfg = self.config
        frame = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
        frame[:, :] = (4, 5, 14)

        star_offset = self.steps % cfg.height
        for x, y, speed in self.stars:
            yy = (int(y) + star_offset * int(speed)) % cfg.height
            brightness = 35 + int(speed) * 32
            frame[yy : yy + 1, x : x + 1] = (brightness, brightness, brightness)

        self._draw_hud(frame)
        for ent in self.energy:
            self._rect(frame, ent, (0, 230, 180))
            self._rect_xy(frame, ent.x + 2, ent.y + 2, ent.w - 4, ent.h - 4, (230, 255, 90))
        for ent in self.meteors:
            self._rect(frame, ent, (220, 64, 42))
            self._rect_xy(frame, ent.x + 2, ent.y + 1, max(1, ent.w - 4), 2, (255, 170, 64))
        for ent in self.enemies:
            self._rect(frame, ent, (170, 70, 255))
            self._rect_xy(frame, ent.x + 2, ent.y + ent.h - 3, ent.w - 4, 2, (70, 245, 255))
        for ent in self.bullets:
            self._rect(frame, ent, (255, 236, 82))

        self._draw_player(frame)
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

    def _spawn_wave(self) -> None:
        cfg = self.config
        difficulty = min(1.0, self.steps / 2_500)
        meteor_chance = 0.055 + 0.045 * difficulty
        enemy_chance = 0.008 + 0.018 * difficulty
        energy_chance = 0.012

        if self.rng.random() < meteor_chance and len(self.meteors) < 12:
            size = int(self.rng.integers(6, 15))
            self.meteors.append(
                Entity(
                    x=float(self.rng.integers(4, cfg.width - size - 4)),
                    y=-size,
                    w=size,
                    h=size,
                    vx=float(self.rng.uniform(-0.35, 0.35)),
                    vy=float(self.rng.uniform(1.0, 2.6 + difficulty)),
                    kind="meteor",
                )
            )

        if self.rng.random() < energy_chance and len(self.energy) < 4:
            self.energy.append(
                Entity(
                    x=float(self.rng.integers(5, cfg.width - 11)),
                    y=-8,
                    w=7,
                    h=7,
                    vx=0,
                    vy=float(self.rng.uniform(1.1, 2.1)),
                    kind="energy",
                )
            )

        if self.rng.random() < enemy_chance and len(self.enemies) < 4:
            self.enemies.append(
                Entity(
                    x=float(self.rng.integers(8, cfg.width - 18)),
                    y=-10,
                    w=12,
                    h=8,
                    vx=float(self.rng.choice([-1.25, 1.25])),
                    vy=float(self.rng.uniform(0.8, 1.7)),
                    kind="enemy",
                )
            )

    def _move_entities(self) -> None:
        cfg = self.config
        for group in (self.bullets, self.meteors, self.energy, self.enemies):
            for ent in group:
                ent.x += ent.vx
                ent.y += ent.vy
                if ent.kind == "enemy" and (ent.x <= 2 or ent.x + ent.w >= cfg.width - 2):
                    ent.vx *= -1

    def _resolve_bullet_hits(self) -> float:
        reward = 0.0
        remaining_bullets: list[Entity] = []
        destroyed_meteors: set[int] = set()
        destroyed_enemies: set[int] = set()

        for bullet in self.bullets:
            hit = False
            for idx, meteor in enumerate(self.meteors):
                if idx not in destroyed_meteors and intersects(bullet, meteor):
                    destroyed_meteors.add(idx)
                    self.score += 25
                    reward += 2.0
                    hit = True
                    break
            if hit:
                continue
            for idx, enemy in enumerate(self.enemies):
                if idx not in destroyed_enemies and intersects(bullet, enemy):
                    destroyed_enemies.add(idx)
                    self.score += 50
                    reward += 3.0
                    hit = True
                    break
            if not hit:
                remaining_bullets.append(bullet)

        self.bullets = remaining_bullets
        self.meteors = [m for idx, m in enumerate(self.meteors) if idx not in destroyed_meteors]
        self.enemies = [e for idx, e in enumerate(self.enemies) if idx not in destroyed_enemies]
        return reward

    def _resolve_player_hits(self) -> float:
        reward = 0.0
        hit_meteors: set[int] = set()
        hit_energy: set[int] = set()
        hit_enemies: set[int] = set()

        for idx, meteor in enumerate(self.meteors):
            if intersects(self.player, meteor):
                hit_meteors.add(idx)
                self.lives -= 1
                reward -= 5.0

        for idx, energy in enumerate(self.energy):
            if intersects(self.player, energy):
                hit_energy.add(idx)
                self.score += 10
                reward += 1.5
                if self.lives < self.config.starting_lives:
                    self.lives += 1
                    reward += 1.0

        for idx, enemy in enumerate(self.enemies):
            if intersects(self.player, enemy):
                hit_enemies.add(idx)
                self.lives -= 1
                reward -= 6.0

        self.meteors = [m for idx, m in enumerate(self.meteors) if idx not in hit_meteors]
        self.energy = [e for idx, e in enumerate(self.energy) if idx not in hit_energy]
        self.enemies = [e for idx, e in enumerate(self.enemies) if idx not in hit_enemies]
        return reward

    def _remove_offscreen(self) -> float:
        cfg = self.config
        reward = 0.0
        passed = 0

        fresh_meteors = []
        for meteor in self.meteors:
            if meteor.y <= cfg.height:
                fresh_meteors.append(meteor)
            else:
                passed += 1
        self.meteors = fresh_meteors

        if passed:
            self.score += passed
            reward += 0.1 * passed

        self.energy = [e for e in self.energy if e.y <= cfg.height]
        self.enemies = [e for e in self.enemies if e.y <= cfg.height]
        self.bullets = [b for b in self.bullets if b.y + b.h >= 0]
        return reward

    def _draw_hud(self, frame: np.ndarray) -> None:
        cfg = self.config
        self._rect_xy(frame, 0, 0, cfg.width, 7, (12, 18, 34))
        for idx in range(self.lives):
            x = 4 + idx * 8
            self._rect_xy(frame, x, 2, 5, 3, (60, 255, 120))

        cooldown_width = int((1 - self.cooldown / cfg.fire_cooldown) * 26)
        self._rect_xy(frame, cfg.width - 32, 2, 26, 3, (45, 45, 70))
        self._rect_xy(frame, cfg.width - 32, 2, cooldown_width, 3, (255, 230, 90))

    def _draw_player(self, frame: np.ndarray) -> None:
        p = self.player
        self._rect_xy(frame, p.x + 2, p.y, p.w - 4, 2, (180, 255, 255))
        self._rect_xy(frame, p.x + 1, p.y + 2, p.w - 2, 4, (38, 205, 255))
        self._rect_xy(frame, p.x, p.y + 5, p.w, 3, (40, 255, 120))

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
