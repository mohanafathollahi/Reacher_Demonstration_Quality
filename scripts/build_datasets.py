

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_IDS = {"sac": 0, "ppo": 1}
VARIANTS = ("high_quality", "low_quality", "random_baseline")


def load_trajectories(source_root):
    """Load both sources and map environment reset seeds to episode IDs."""
    datasets = {}

    for source in SOURCE_IDS:
        folder = Path(source_root) / source

        meta_path = folder / "episode_metadata.csv"
        traj_path = folder / "trajectories.npz"

        if not meta_path.is_file() or not traj_path.is_file():
            raise FileNotFoundError(
                f"Missing collection for {source}: "
                f"expected {meta_path} and {traj_path}"
            )

        meta = pd.read_csv(meta_path)

        required = {"episode_id", "reset_seed"}

        if not required.issubset(meta.columns):
            raise ValueError(
                f"{meta_path} needs columns {sorted(required)}"
            )

        if meta[list(required)].isna().any().any():
            raise ValueError(
                f"Missing episode ID or reset seed in {meta_path}"
            )

        if meta["episode_id"].duplicated().any():
            raise ValueError(f"Duplicate episode_id for {source}")

        if meta["reset_seed"].duplicated().any():
            raise ValueError(f"Duplicate reset_seed for {source}")

        with np.load(traj_path, allow_pickle=False) as data:
            for key in ("observations", "actions", "episode_ids"):
                if key not in data:
                    raise ValueError(f"Missing {key} in {traj_path}")

            obs = data["observations"].copy()
            actions = data["actions"].copy()
            episode_ids = data["episode_ids"].copy()

        if not (len(obs) == len(actions) == len(episode_ids)) or not len(obs):
            raise ValueError(
                f"Inconsistent or empty transition arrays for {source}")

        if set(meta["episode_id"]) != set(
            np.unique(episode_ids).tolist()):
            raise ValueError(
                f"Trajectory episode IDs do not match metadata for {source}")

        if obs.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                f"Expected 2D observations/actions for {source}")

        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError(f"Non-finite observations/actions in {source}")

        datasets[source] = {
            "meta": meta,
            "obs": obs,
            "actions": actions,
            "episode_ids": episode_ids,
            "seed_to_id": dict(zip(meta["reset_seed"].astype(int),
                                                  meta["episode_id"])),}

    if (datasets["sac"]["obs"].shape[1]
        != datasets["ppo"]["obs"].shape[1]):
        raise ValueError("SAC and PPO have different observation dimensions")

    if (datasets["sac"]["actions"].shape[1]
        != datasets["ppo"]["actions"].shape[1]):
        raise ValueError("SAC and PPO have different action dimensions")

    sac_seeds = set(datasets["sac"]["seed_to_id"])
    ppo_seeds = set(datasets["ppo"]["seed_to_id"])

    if sac_seeds != ppo_seeds:
        raise ValueError(
            f"SAC and PPO must contain matching reset_seed values; "
            f"{len(sac_seeds - ppo_seeds)} only in SAC and "
            f"{len(ppo_seeds - sac_seeds)} only in PPO.")

    return datasets, np.asarray(sorted(sac_seeds), dtype=np.int64)


def allocate_seeds(all_seeds, sizes, val_fraction, rng):
    """Make nested, globally disjoint train/validation seed selections."""

    max_size = max(sizes)
    if max_size > len(all_seeds):
        raise ValueError(f"Need {max_size} paired tasks, found {len(all_seeds)}")
    pool_val_n = max(2, int(round(len(all_seeds) * val_fraction)))

    if pool_val_n % 2:
        pool_val_n += 1
    if pool_val_n >= len(all_seeds):
        raise ValueError("The global validation pool would leave no training tasks")

    shuffled = rng.permutation(all_seeds)

    global_val = shuffled[:pool_val_n]
    global_train = shuffled[pool_val_n:]

    selections = {}

    for n in sorted(sizes, reverse=True):

        if n < 4 or n % 2:
            raise ValueError(
                f"Size {n} must be even and at least 4"
            )

        n_val = int(round(n * val_fraction))
        n_train = n - n_val

        if (
            n_val < 2
            or n_train < 2
            or n_val % 2
            or n_train % 2
        ):
            raise ValueError(
                f"For size {n}, the train and validation episode counts "
                f"({n_train}, {n_val}) must both be positive even numbers "
                "to preserve a 50/50 mixture in both splits."
            )

        if (
            n_val > len(global_val)
            or n_train > len(global_train)
        ):
            raise ValueError(
                f"Not enough globally held-out tasks for size {n}: "
                f"need {n_train} train and {n_val} val; "
                f"have {len(global_train)} and {len(global_val)}."
            )

        selections[n] = {
            "train": global_train[:n_train].copy(),
            "val": global_val[:n_val].copy(),
        }

    return selections


def select_episodes(datasets, seeds, rng):
    """For one split, create SAC, PPO and exactly balanced mixed selections."""

    # Both teachers cover the same environment reset seeds.
    seeds = np.asarray(seeds)

    if len(seeds) % 2:
        raise ValueError(
            "Balanced random_baseline requires an even episode count"
        )

    half = len(seeds) // 2

    sac_mixed_seeds = rng.choice(seeds,size=half,replace=False)

    ppo_mixed_seeds = rng.choice(seeds,
                                size=half,
                                replace=False
                            )

    def ids(source, chosen_seeds):
        mapping = datasets[source]["seed_to_id"]

        return np.asarray([
            mapping[int(s)]
            for s in chosen_seeds
        ])

    return {
        "high_quality": {
            "sac": ids("sac", seeds),
            "ppo": np.array([], dtype=np.int64)
        },

        "low_quality": {
            "sac": np.array([], dtype=np.int64),
            "ppo": ids("ppo", seeds)
        },

        "random_baseline": {
            "sac": ids("sac", sac_mixed_seeds),
            "ppo": ids("ppo", ppo_mixed_seeds),
        },
    }


def save_split(datasets, selected, output_path):
    """Save BC compatible arrays; identifiers distinguish SAC/PPO episodes."""

    observations = []
    actions = []
    episode_ids = []
    original_ids = []
    model_ids = []

    episode_count = 0

    for source, source_num in SOURCE_IDS.items():

        ids = selected[source]

        if not len(ids):
            continue

        data = datasets[source]

        mask = np.isin(
            data["episode_ids"],
            ids
        )

        if not mask.any():
            raise ValueError(
                f"No transitions for {source} selected episodes"
            )

        selected_ids = data["episode_ids"][mask]

        episode_map = {
            int(eid): episode_count + i
            for i, eid in enumerate(ids)
        }

        observations.append(
            data["obs"][mask]
        )

        actions.append(
            data["actions"][mask]
        )

        original_ids.append(selected_ids)

        episode_ids.append(
            np.asarray(
                [ episode_map[int(eid)]
                    for eid in selected_ids
                ],
                dtype=np.int32
            )
        )

        model_ids.append(
            np.full(
                len(selected_ids),
                source_num,
                dtype=np.int8
            )
        )

        episode_count += len(ids)

    if not observations:
        raise ValueError(
            "Cannot save an empty dataset"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        episode_ids=np.concatenate(episode_ids),
        source_episode_ids=np.concatenate(original_ids),
        model_ids=np.concatenate(model_ids),
    )

    return episode_count, sum(
        len(x) for x in observations
    )


def build(source_root,
          out,
          sizes,
          val_fraction,
          seed):

    rng = np.random.default_rng(seed)

    datasets, paired_seeds = load_trajectories(source_root)

    selections = allocate_seeds(paired_seeds,
                                sizes,
                                val_fraction,
                                rng)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    records = []

    for n in sizes:
        folder = out / f"episodes_{n}"
        folder.mkdir(parents=True, exist_ok=True)
        for split, eval_seeds in selections[n].items():
            variants = select_episodes(datasets,
                                        eval_seeds,
                                        rng)
            for variant in VARIANTS:
                choice = variants[variant]
                path = folder / f"{variant}_{split}.npz"
                count, transitions = save_split(datasets,
                                                choice,
                                                path)

                records.append({
                    "variation": n,
                    "variant": variant,
                    "split": split,
                    "episodes": count,
                    "sac_episodes": len(choice["sac"]),
                    "ppo_episodes": len(choice["ppo"]),
                    "transitions": transitions,
                })

        pd.DataFrame([ r for r in records if r["variation"] == n]).to_csv(folder / "manifest.csv",index=False)
    manifest = pd.DataFrame(records)
    manifest.to_csv(out / "manifest.csv",index=False)

    print(manifest.to_string(index=False))

    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/expert"))
    parser.add_argument("--out", type=Path, default=Path("data/variants"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[300, 150])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")

    if len(set(args.sizes)) != len(args.sizes):
        parser.error("--sizes must be unique")

    build(args.source,
        args.out,
        args.sizes,
        args.val_fraction,
        args.seed
    )


if __name__ == "__main__":
    main()