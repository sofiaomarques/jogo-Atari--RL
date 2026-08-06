from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .env import ACTIONS

StateKey = tuple[int, ...]


@dataclass(frozen=True)
class StateDiscretizer:
    paddle_bins: int = 12
    ball_x_bins: int = 12
    ball_y_bins: int = 10
    rel_edges: tuple[float, ...] = (-0.45, -0.22, -0.08, 0.08, 0.22, 0.45)

    @property
    def state_size(self) -> int:
        return 7

    def discretize(self, observation: Iterable[float] | np.ndarray) -> StateKey:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] < 8:
            raise ValueError("state observation must have 8 values")

        paddle_x = float(obs[0])
        ball_x = float(obs[1])
        ball_y = float(obs[2])
        ball_vx = float(obs[3])
        ball_vy = float(obs[4])
        target_delta = float(obs[5])
        streak = float(obs[7])

        return (
            self._uniform_bin(paddle_x, self.paddle_bins),
            self._uniform_bin(ball_x, self.ball_x_bins),
            self._uniform_bin(ball_y, self.ball_y_bins),
            self._edge_bin(ball_vx, (-0.05, 0.05)),
            self._edge_bin(ball_vy, (-0.05, 0.05)),
            self._edge_bin(target_delta, self.rel_edges),
            self._edge_bin(streak, (0.05, 0.25, 0.5, 0.8)),
        )

    @staticmethod
    def _uniform_bin(value: float, bins: int) -> int:
        clipped = min(max(float(value), 0.0), 0.999999)
        return int(clipped * bins)

    @staticmethod
    def _edge_bin(value: float, edges: tuple[float, ...]) -> int:
        return int(np.digitize(float(value), np.asarray(edges, dtype=np.float32)))


@dataclass
class QLearningAgent:
    action_count: int = len(ACTIONS)
    alpha: float = 0.18
    gamma: float = 0.98
    discretizer: StateDiscretizer = field(default_factory=StateDiscretizer)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    q_table: dict[StateKey, np.ndarray] = field(default_factory=dict)

    def act(self, observation: Iterable[float] | np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        values = self.q_values(observation)
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def learn(
        self,
        observation: Iterable[float] | np.ndarray,
        action: int,
        reward: float,
        next_observation: Iterable[float] | np.ndarray,
        done: bool,
    ) -> None:
        values = self.q_values(observation)
        next_values = self.q_values(next_observation)
        future = 0.0 if done else float(next_values.max())
        target = float(reward) + self.gamma * future
        values[int(action)] += self.alpha * (target - values[int(action)])

    def q_values(self, observation: Iterable[float] | np.ndarray) -> np.ndarray:
        key = self.discretizer.discretize(observation)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_count, dtype=np.float32)
        return self.q_table[key]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.q_table.keys()), dtype=np.int16)
        values = np.array(list(self.q_table.values()), dtype=np.float32)
        if keys.size == 0:
            keys = np.empty((0, self.discretizer.state_size), dtype=np.int16)
        if values.size == 0:
            values = np.empty((0, self.action_count), dtype=np.float32)

        metadata = {
            "action_count": self.action_count,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        np.savez_compressed(
            path,
            keys=keys,
            values=values,
            metadata=np.array(json.dumps(metadata)),
        )

    @classmethod
    def load(cls, path: str | Path, rng: np.random.Generator | None = None) -> QLearningAgent:
        data = np.load(Path(path), allow_pickle=False)
        metadata = json.loads(str(data["metadata"].item()))
        agent = cls(
            action_count=int(metadata["action_count"]),
            alpha=float(metadata["alpha"]),
            gamma=float(metadata["gamma"]),
            rng=rng or np.random.default_rng(),
        )
        agent.q_table = {
            tuple(int(part) for part in key): value.astype(np.float32)
            for key, value in zip(data["keys"], data["values"])
        }
        return agent
