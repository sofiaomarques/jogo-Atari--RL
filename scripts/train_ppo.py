from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game import AtariDodgeEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent with Stable-Baselines3.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--obs", choices=["state", "pixels", "rgb"], default="state")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("models/neon_intercept_ppo"))
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Stable-Baselines3 is required for training. "
            "Install optional dependencies with: pip install gymnasium stable-baselines3"
        ) from exc

    def make_env() -> Monitor:
        return Monitor(AtariDodgeEnv(obs_mode=args.obs))

    env = DummyVecEnv([make_env])
    policy = "MlpPolicy"
    if args.obs in {"pixels", "rgb"}:
        env = VecTransposeImage(env)
        policy = "CnnPolicy"

    model = PPO(
        policy,
        env,
        verbose=1,
        seed=args.seed,
        n_steps=512,
        batch_size=64,
        learning_rate=2.5e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.timesteps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
