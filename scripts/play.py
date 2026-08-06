from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game.game import NeonInterceptGame
from atari_rl_game.tk_renderer import draw_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Atari paddle-ball game.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    game = NeonInterceptGame()
    game.reset(seed=args.seed)
    scale = 3
    pressed: set[str] = set()

    root = tk.Tk()
    root.title("Atari Paddle RL")
    root.resizable(False, False)

    cfg = game.config
    canvas = tk.Canvas(
        root,
        width=cfg.width * scale,
        height=cfg.height * scale,
        bg="#04050e",
        highlightthickness=0,
    )
    canvas.pack()

    terminated = False
    truncated = False

    def reset() -> None:
        nonlocal terminated, truncated
        game.reset(seed=args.seed)
        terminated = False
        truncated = False

    def key_down(event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "r":
            reset()
        else:
            pressed.add(key)

    def key_up(event: tk.Event) -> None:
        pressed.discard(event.keysym.lower())

    def action_from_keys() -> int:
        if "left" in pressed:
            return 1
        if "right" in pressed:
            return 2
        return 0

    def tick() -> None:
        nonlocal terminated, truncated
        if not (terminated or truncated):
            _, terminated, truncated, _ = game.step(action_from_keys())
        overlay = "FIM - aperte R" if terminated or truncated else None
        draw_game(canvas, game, scale=scale, overlay=overlay)
        root.title(
            f"Atari Paddle RL | score {game.score} | vidas {game.lives} | rebatidas {game.hits}"
        )
        root.after(33, tick)

    root.bind("<KeyPress>", key_down)
    root.bind("<KeyRelease>", key_up)
    root.after(0, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
