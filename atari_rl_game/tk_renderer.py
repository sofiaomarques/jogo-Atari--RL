from __future__ import annotations

import tkinter as tk

from .game import NeonInterceptGame


def draw_game(
    canvas: tk.Canvas,
    game: NeonInterceptGame,
    scale: int = 3,
    overlay: str | None = None,
) -> None:
    cfg = game.config

    def rect(x: float, y: float, w: float, h: float, color: str) -> None:
        canvas.create_rectangle(
            x * scale,
            y * scale,
            (x + w) * scale,
            (y + h) * scale,
            fill=color,
            outline=color,
        )

    canvas.delete("all")
    rect(0, 0, cfg.width, cfg.height, "#04050e")

    star_offset = game.steps % cfg.height
    for x, y, speed in game.stars:
        yy = (int(y) + star_offset * int(speed)) % cfg.height
        shade = 35 + int(speed) * 34
        rect(int(x), yy, 1, 1, f"#{shade:02x}{shade:02x}{shade:02x}")

    rect(0, 0, cfg.width, 8, "#0c1222")
    for idx in range(game.lives):
        rect(4 + idx * 8, 2, 5, 3, "#3cff78")

    streak_width = int(min(game.streak, 20) / 20 * 42)
    rect(cfg.width - 48, 2, 42, 3, "#2d2d46")
    rect(cfg.width - 48, 2, streak_width, 3, "#ffe65a")
    rect(0, game.paddle.y + game.paddle.h + 7, cfg.width, 1, "#192236")
    rect(game.paddle.x, game.paddle.y, game.paddle.w, game.paddle.h, "#28ff78")
    rect(game.paddle.x + 2, game.paddle.y - 1, game.paddle.w - 4, 2, "#b4ffff")
    rect(game.ball.x, game.ball.y, game.ball.w, game.ball.h, "#ffec52")
    rect(game.ball.x + 1, game.ball.y + 1, 2, 2, "#ffffff")

    if overlay:
        canvas.create_text(
            cfg.width * scale / 2,
            cfg.height * scale / 2,
            text=overlay,
            fill="#ffffff",
            font=("Menlo", 20, "bold"),
        )
