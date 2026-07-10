from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game.game import NeonInterceptGame


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Neon Intercept with the keyboard.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    game = NeonInterceptGame()
    game.reset(seed=args.seed)
    scale = 3
    pressed: set[str] = set()

    root = tk.Tk()
    root.title("Neon Intercept RL")
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
        fire = "space" in pressed
        if "left" in pressed:
            return 6 if fire else 1
        if "right" in pressed:
            return 7 if fire else 2
        if "up" in pressed:
            return 3
        if "down" in pressed:
            return 4
        return 5 if fire else 0

    def rect(x: float, y: float, w: float, h: float, color: str) -> None:
        canvas.create_rectangle(
            x * scale,
            y * scale,
            (x + w) * scale,
            (y + h) * scale,
            fill=color,
            outline=color,
        )

    def draw() -> None:
        canvas.delete("all")
        rect(0, 0, cfg.width, cfg.height, "#04050e")

        star_offset = game.steps % cfg.height
        for x, y, speed in game.stars:
            yy = (int(y) + star_offset * int(speed)) % cfg.height
            shade = 35 + int(speed) * 32
            color = f"#{shade:02x}{shade:02x}{shade:02x}"
            rect(int(x), yy, 1, 1, color)

        rect(0, 0, cfg.width, 7, "#0c1222")
        for idx in range(game.lives):
            rect(4 + idx * 8, 2, 5, 3, "#3cff78")

        cooldown_width = int((1 - game.cooldown / cfg.fire_cooldown) * 26)
        rect(cfg.width - 32, 2, 26, 3, "#2d2d46")
        rect(cfg.width - 32, 2, cooldown_width, 3, "#ffe65a")

        for ent in game.energy:
            rect(ent.x, ent.y, ent.w, ent.h, "#00e6b4")
            rect(ent.x + 2, ent.y + 2, ent.w - 4, ent.h - 4, "#e6ff5a")
        for ent in game.meteors:
            rect(ent.x, ent.y, ent.w, ent.h, "#dc402a")
            rect(ent.x + 2, ent.y + 1, max(1, ent.w - 4), 2, "#ffaa40")
        for ent in game.enemies:
            rect(ent.x, ent.y, ent.w, ent.h, "#aa46ff")
            rect(ent.x + 2, ent.y + ent.h - 3, ent.w - 4, 2, "#46f5ff")
        for ent in game.bullets:
            rect(ent.x, ent.y, ent.w, ent.h, "#ffec52")

        p = game.player
        rect(p.x + 2, p.y, p.w - 4, 2, "#b4ffff")
        rect(p.x + 1, p.y + 2, p.w - 2, 4, "#26cdff")
        rect(p.x, p.y + 5, p.w, 3, "#28ff78")

        if terminated or truncated:
            canvas.create_text(
                cfg.width * scale / 2,
                cfg.height * scale / 2,
                text="FIM - aperte R",
                fill="#ffffff",
                font=("Menlo", 20, "bold"),
            )

        root.title(
            f"Neon Intercept RL | score {game.score} | vidas {game.lives} | passos {game.steps}"
        )

    def tick() -> None:
        nonlocal terminated, truncated
        if not (terminated or truncated):
            _, terminated, truncated, _ = game.step(action_from_keys())
        draw()
        root.after(33, tick)

    root.bind("<KeyPress>", key_down)
    root.bind("<KeyRelease>", key_up)
    root.after(0, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
