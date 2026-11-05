"""Shared Reacher-v5 rollout metrics, model architecture, and CSV helpers."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

ENV_ID = "Reacher-v5"
OBS_DIM = 10
ACT_DIM = 2


def env():
    import gymnasium as gym
    return gym.make(ENV_ID)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, ACT_DIM), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def policy_rollout(policy,
                   seed):
    """Evaluate a deterministic BC policy on a fresh environment and seed."""
    e = env()
    try:
        obs, _ = e.reset(seed=int(seed))
        distances, costs, actions = [], [], []
        episode_return = 0.0
        while True:
            with torch.no_grad():
                action = policy(torch.as_tensor(obs, dtype=torch.float32)).cpu().numpy()
            obs, reward, terminated, truncated, info = e.step(action)
            episode_return += float(reward)
            distances.append(-float(info['reward_dist']))
            costs.append(-float(info['reward_ctrl']))
            actions.append(np.asarray(action, dtype=np.float32))
            if terminated or truncated:
                break
        changes = np.diff(np.stack(actions), axis=0)
        return dict(episode_return=episode_return,
                    mean_distance=float(np.mean(distances)),
                    final_distance=float(distances[-1]),
                    min_distance=float(np.min(distances)),
                    success_2cm=int(np.min(distances) < 0.02),
                    mean_control_cost=float(np.mean(costs)),
                    mean_action_change=float(np.linalg.norm(changes, axis=1).mean()) if len(changes) else 0.0)
    finally:
        e.close()


def load_bc(path):
    # Only load checkpoints you trust. weights_only avoids arbitrary pickle objects.
    policy = BCPolicy()
    policy.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    policy.eval()
    return policy