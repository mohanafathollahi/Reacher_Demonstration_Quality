"""Collect full deterministic SAC trajectories. Use seeds disjoint from final test."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from src.common import env


def collect(model_path,
            output,
            episodes,
            seed_start):

    model = SAC.load(str(model_path), device='cpu')
    e = env()
    observations, actions, episode_ids = [], [], []
    metadata = []
    try:
        for i in range(episodes):
            reset_seed = seed_start + i
            obs, _ = e.reset(seed=reset_seed)
            distances, costs, returns = [], [], 0.0
            n = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(np.asarray(action, dtype=np.float32).copy())
                episode_ids.append(i)
                obs, reward, terminated, truncated, info = e.step(action)
                distances.append(-float(info['reward_dist']))
                costs.append(-float(info['reward_ctrl']))
                returns += float(reward)
                n += 1
                if terminated or truncated:
                    break
            metadata.append(dict(episode_id=i, reset_seed=reset_seed, length=n,
                                 episode_return=returns,
                                 mean_distance=float(np.mean(distances)),
                                 final_distance=float(distances[-1]),
                                 min_distance=float(np.min(distances)),
                                 success_2cm=int(np.min(distances) < 0.02),
                                 mean_control_cost=float(np.mean(costs))))
    finally:
        e.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'trajectories.npz',
                        observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                        episode_ids=np.asarray(episode_ids, dtype=np.int32))
    pd.DataFrame(metadata).to_csv(output / 'episode_metadata.csv', index=False)
    print(f'Collected {episodes} episodes, {len(actions)} transitions at {output}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True, help='Existing SAC .zip checkpoint')
    p.add_argument('--out', type=Path, default=Path('data/expert'))
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--seed-start', type=int, default=10000)
    a = p.parse_args()
    if a.episodes < 10 or a.seed_start < 0:
        p.error('Need at least 10 episodes and a nonnegative reset seed.')
    collect(a.model,
            a.out,
            a.episodes,
            a.seed_start)

if __name__ == '__main__':
    main()
