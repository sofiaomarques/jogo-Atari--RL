from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atari_rl_game import AtariDodgeEnv
from atari_rl_game.game import GameConfig
from atari_rl_game.q_learning import QLearningAgent


def epsilon_for_episode(episode: int, episodes: int, start: float, end: float) -> float:
    if episodes <= 1:
        return end
    progress = (episode - 1) / (episodes - 1)
    return end + (start - end) * (1.0 - progress)


def run_episode(
    env: AtariDodgeEnv,
    agent: QLearningAgent,
    seed: int,
    epsilon: float,
    train: bool,
) -> tuple[float, int, int, int]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = agent.act(observation, epsilon=epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        if train:
            agent.learn(
                observation,
                action,
                reward,
                next_observation,
                terminated or truncated,
            )
        observation = next_observation
        total_reward += reward

    return total_reward, int(info["score"]), int(info["hits"]), int(info["steps"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-learning to keep the ball from falling.")
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.03)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("models/q_policy.npz"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = AtariDodgeEnv(obs_mode="state", config=GameConfig(max_steps=args.max_steps))
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, rng=rng)

    recent_scores: list[int] = []
    recent_hits: list[int] = []
    best_hits = 0

    for episode in range(1, args.episodes + 1):
        epsilon = epsilon_for_episode(
            episode,
            args.episodes,
            args.epsilon_start,
            args.epsilon_end,
        )
        reward, score, hits, steps = run_episode(
            env,
            agent,
            seed=args.seed + episode,
            epsilon=epsilon,
            train=True,
        )
        recent_scores.append(score)
        recent_hits.append(hits)
        recent_scores = recent_scores[-args.report_every :]
        recent_hits = recent_hits[-args.report_every :]
        best_hits = max(best_hits, hits)

        if episode == 1 or episode % args.report_every == 0 or episode == args.episodes:
            print(
                f"episodio={episode:4d} "
                f"exploracao={epsilon:.3f} "
                f"media_pontos={np.mean(recent_scores):7.1f} "
                f"media_rebatidas={np.mean(recent_hits):6.1f} "
                f"rebatidas={hits:4d} "
                f"melhor={best_hits:4d} "
                f"passos={steps:4d} "
                f"estados={len(agent.q_table):5d} "
                f"recompensa={reward:8.2f}"
            )

    agent.save(args.out)
    print(f"Politica salva em: {args.out}")

    if args.eval_episodes > 0:
        scores = []
        hits_list = []
        for idx in range(args.eval_episodes):
            _, score, hits, _ = run_episode(
                env,
                agent,
                seed=args.seed + 10_000 + idx,
                epsilon=0.0,
                train=False,
            )
            scores.append(score)
            hits_list.append(hits)

        print(
            "Avaliacao sem exploracao: "
            f"pontos_medios={np.mean(scores):.1f} "
            f"rebatidas_medias={np.mean(hits_list):.1f} "
            f"melhor_rebatidas={np.max(hits_list):.0f}"
        )


if __name__ == "__main__":
    main()
