from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game import AtariDodgeEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a random agent.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--obs", choices=["pixels", "rgb", "state"], default="state")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = AtariDodgeEnv(
        obs_mode=args.obs,
        render_mode="human" if args.render else None,
    )

    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if args.render:
                    env.render()

            print(
                f"episode={episode + 1} "
                f"reward={total_reward:.2f} "
                f"score={info['score']} "
                f"steps={info['steps']} "
                f"lives={info['lives']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
