"""
Evaluate Behavioral Cloning policies and SAC/PPO teacher models.

All models are evaluated using identical held-out
Reacher-v5 environment seeds.

Supports:
    - Multiple dataset sizes (150, 300)
    - Three BC variants
    - Multiple BC training seeds
    - SAC and PPO teacher baselines
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load_bc, policy_rollout, env


METRICS = [
    "episode_return",
    "mean_distance",
    "final_distance",
    "min_distance",
    "success_2cm",
    "mean_control_cost",
    "mean_action_change",
]


def teacher_rollout(model, episode_seed):
    """
    Evaluate one episode using a pretrained SAC or PPO policy.
    Uses the same metrics as the BC evaluation.
    """
    e = env()

    try:
        obs, _ = e.reset(seed=int(episode_seed))

        distances = []
        costs = []
        actions = []

        episode_return = 0.0

        while True:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = e.step(action)

            episode_return += float(reward)

            distances.append(
                -float(info["reward_dist"])
            )

            costs.append(
                -float(info["reward_ctrl"])
            )

            actions.append(
                np.asarray(action, dtype=np.float32).copy()
            )

            if terminated or truncated:
                break

        changes = np.diff(
            np.stack(actions),
            axis=0
        )

        return {
            "episode_return": episode_return,
            "mean_distance": float(np.mean(distances)),
            "final_distance": float(distances[-1]),
            "min_distance": float(np.min(distances)),
            "success_2cm": int(np.min(distances) < 0.02),
            "mean_control_cost": float(np.mean(costs)),
            "mean_action_change": (
                float(np.linalg.norm(changes, axis=1).mean())
                if len(changes)
                else 0.0
            ),
        }

    finally:
        e.close()


def evaluate(
    models,
    sizes,
    variants,
    seeds,
    evaluation_seeds,
    sac_teacher=None,
    ppo_teacher=None
):
    """
    Evaluate BC models across all dataset sizes,
    followed by SAC and PPO teachers.
    """
    if not evaluation_seeds:
        raise ValueError("At least one evaluation seed is required.")

    rows = []

    # ----------------------------------
    # 1. Evaluate Behavioral Cloning
    # ----------------------------------

    for size in sizes:

        for variant in variants:

            for train_seed in seeds:

                model_path = (
                    models
                    / f"episodes_{size}"
                    / f"{variant}_seed{train_seed}.pt"
                )

                if not model_path.is_file():
                    raise FileNotFoundError(
                        f"Missing BC model: {model_path}"
                    )

                policy = load_bc(model_path)

                for eval_seed in evaluation_seeds:

                    result = policy_rollout(
                        policy,
                        eval_seed
                    )

                    rows.append({
                        "dataset_size": size,
                        "variant": variant,
                        "training_seed": train_seed,
                        "evaluation_seed": eval_seed,
                        "model_type": "bc",
                        "teacher_source": "",
                        **result,
                    })

                print(
                    f"Evaluated BC {variant} | "
                    f"size={size} | "
                    f"seed={train_seed} | "
                    f"episodes={len(evaluation_seeds)}",
                    flush=True
                )

    # ----------------------------------
    # 2. Evaluate SAC and PPO teachers
    # ----------------------------------

    teacher_paths = {
        "sac_teacher": sac_teacher,
        "ppo_teacher": ppo_teacher,
    }

    if any(path is not None for path in teacher_paths.values()):

        from stable_baselines3 import SAC, PPO

        algorithms = {
            "sac_teacher": SAC,
            "ppo_teacher": PPO,
        }

        for teacher_name, model_path in teacher_paths.items():

            if model_path is None:
                continue

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Missing teacher model: {model_path}"
                )

            model = algorithms[teacher_name].load(
                str(model_path),
                device="cpu"
            )

            for eval_seed in evaluation_seeds:

                result = teacher_rollout(
                    model,
                    eval_seed
                )

                rows.append({
                    "dataset_size": -1,
                    "variant": teacher_name,
                    "training_seed": -1,
                    "evaluation_seed": eval_seed,
                    "model_type": "teacher",
                    "teacher_source": teacher_name.replace(
                        "_teacher", ""
                    ),
                    **result,
                })

            print(
                f"Evaluated {teacher_name} | "
                f"episodes={len(evaluation_seeds)}",
                flush=True
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC and teacher policies.")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[150, 300])
    parser.add_argument("--variants", nargs="+", default=["high_quality", "low_quality", "random_baseline"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--test-seed-start", type=int, default=20000)
    parser.add_argument("--sac-teacher", type=Path, default=None)
    parser.add_argument("--ppo-teacher", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/evaluation.csv"))

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")

    if not args.sizes or any(size < 1 for size in args.sizes):
        parser.error("--sizes must contain positive integers")

    if len(set(args.sizes)) != len(args.sizes):
        parser.error("--sizes must not contain duplicates")

    if not args.seeds or not args.variants:
        parser.error("--seeds and --variants must be nonempty")

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    evaluation_seeds = list(
        range(
            args.test_seed_start,
            args.test_seed_start + args.episodes
        )
    )

    df = evaluate(
        models=args.models,
        sizes=args.sizes,
        variants=args.variants,
        seeds=args.seeds,
        evaluation_seeds=evaluation_seeds,
        sac_teacher=args.sac_teacher,
        ppo_teacher=args.ppo_teacher
    )

    # ----------------------------------
    # 3. Save episode-level results
    # ----------------------------------

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        args.out,
        index=False
    )

    # ----------------------------------
    # 4. Calculate model-level averages
    # ----------------------------------

    model_summary = (
        df.groupby(
            [
                "dataset_size",
                "variant",
                "training_seed",
                "model_type"
            ],
            as_index=False
        )[METRICS]
        .mean()
    )

    summary_path = args.out.with_name(
        "model_summary.csv"
    )

    model_summary.to_csv(
        summary_path,
        index=False
    )

    print("\nEvaluation completed.")

    print(
        f"Saved {len(df)} evaluation episodes to {args.out}"
    )

    print(
        f"Saved model summary to {summary_path}"
    )

    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()