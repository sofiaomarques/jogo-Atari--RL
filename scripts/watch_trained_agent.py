from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tkinter as tk

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game.env import ACTIONS
from atari_rl_game.game import GameConfig, NeonInterceptGame
from atari_rl_game.q_learning import QLearningAgent
from atari_rl_game.tk_renderer import draw_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the trained policy play by itself.")
    parser.add_argument("--model", type=Path, default=Path("models/q_policy.npz"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Modelo nao encontrado: {args.model}\n"
            "Treine primeiro com: python3 scripts/train_q_learning.py --episodes 1000"
        )

    agent = QLearningAgent.load(args.model, rng=np.random.default_rng(args.seed))
    game = NeonInterceptGame(GameConfig(max_steps=args.max_steps))
    game.reset(seed=args.seed)

    scale = 3
    cfg = game.config
    root = tk.Tk()
    root.title("Atari Paddle RL - agente treinado")
    root.resizable(False, False)
    canvas = tk.Canvas(
        root,
        width=cfg.width * scale,
        height=cfg.height * scale,
        bg="#04050e",
        highlightthickness=0,
    )
    canvas.pack()

    episode = 1
    finished = False
    finish_ticks = 0
    last_action = 0

    def reset() -> None:
        nonlocal episode, finished, finish_ticks, last_action
        episode += 1
        finished = False
        finish_ticks = 0
        last_action = 0
        game.reset(seed=args.seed + episode)

    def key_down(event: tk.Event) -> None:
        if event.keysym.lower() == "r":
            reset()

    def tick() -> None:
        nonlocal finished, finish_ticks, last_action
        overlay = None
        if not finished:
            observation = game.state_vector()
            last_action = agent.act(observation, epsilon=args.epsilon)
            _, terminated, truncated, _ = game.step(last_action)
            finished = terminated or truncated
        else:
            finish_ticks += 1
            overlay = "FIM - reiniciando"
            if finish_ticks >= 60:
                reset()

        draw_game(canvas, game, scale=scale, overlay=overlay)
        root.title(
            "Atari Paddle RL - agente treinado | "
            f"ep {episode} | score {game.score} | rebatidas {game.hits} | "
            f"acao {ACTIONS.get(last_action, last_action)}"
        )
        root.after(33, tick)

    root.bind("<KeyPress>", key_down)
    root.after(0, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
