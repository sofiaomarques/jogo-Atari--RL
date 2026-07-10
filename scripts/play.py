from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game import AtariDodgeEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Neon Intercept with the keyboard.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        import pygame
    except ModuleNotFoundError as exc:
        raise SystemExit("pygame is required. Install it with: pip install pygame") from exc

    env = AtariDodgeEnv(obs_mode="pixels", render_mode="human")
    env.reset(seed=args.seed)

    running = True
    terminated = truncated = False
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset(seed=args.seed)
                    terminated = truncated = False

            keys = pygame.key.get_pressed()
            fire = keys[pygame.K_SPACE]
            if keys[pygame.K_LEFT]:
                action = 6 if fire else 1
            elif keys[pygame.K_RIGHT]:
                action = 7 if fire else 2
            elif keys[pygame.K_UP]:
                action = 3
            elif keys[pygame.K_DOWN]:
                action = 4
            else:
                action = 5 if fire else 0

            if not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(action)

            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
