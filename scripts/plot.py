
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

VARIANTS = (
    "high_quality",
    "low_quality",
    "random_baseline",
)

LABELS = {
    "high_quality": "BC · SAC data",
    "low_quality": "BC · PPO data",
    "random_baseline": "BC · 50/50 mix",
    "sac_teacher": "SAC teacher",
    "ppo_teacher": "PPO teacher",
}

METRICS = {
    "episode_return": (
        "Episode return (higher is better)",
        "higher",
    ),
    "mean_distance": (
        "Mean fingertip distance (m; lower is better)",
        "lower",
    ),

    "mean_control_cost": (
        "Mean control cost (lower is better)",
        "lower",
    ),

}

COLORS = {
    "high_quality": "tab:blue",
    "low_quality": "tab:orange",
    "random_baseline": "tab:green",
    "sac_teacher": "#36558f",
    "ppo_teacher": "#a14e25",
}

MARKERS = {
    "high_quality": "o",
    "low_quality": "s",
    "random_baseline": "^",
}


# ---------------------------------------------------------
# 1. Load and validate evaluation results
# ---------------------------------------------------------

def read_summary(path):

    df = pd.read_csv(path)

    required = {
        "dataset_size",
        "variant",
        "training_seed",
        "model_type",
        *METRICS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Summary missing columns: {sorted(missing)}. "
            "Run the updated evaluate.py."
        )

    if df.empty:
        raise ValueError("Summary is empty.")

    for col in (
        "dataset_size",
        "training_seed",
        *METRICS,
    ):
        df[col] = pd.to_numeric(
            df[col],
            errors="raise",
        )

    if not np.isfinite(
        df[list(METRICS)].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite metric values in summary."
        )

    bc = df[
        df.model_type.eq("bc")
    ].copy()

    teacher = df[
        df.model_type.eq("teacher")
    ].copy()

    if bc.empty:
        raise ValueError(
            "No BC rows in summary."
        )

    if bc.duplicated(
        ["dataset_size", "variant", "training_seed"]
    ).any():
        raise ValueError(
            "Duplicate BC model summaries found."
        )

    if teacher.duplicated(["variant"]).any():
        raise ValueError(
            "Duplicate teacher rows found; "
            "one row per teacher is expected."
        )

    unexpected = set(bc.variant) - set(VARIANTS)

    if unexpected:
        raise ValueError(
            f"Unknown BC variants: {sorted(unexpected)}"
        )

    if (bc.dataset_size <= 0).any():
        raise ValueError(
            "BC dataset_size must be positive."
        )

    return bc, teacher


# ---------------------------------------------------------
# 2. Add SAC and PPO teacher reference lines
# ---------------------------------------------------------

def teacher_reference(ax, teacher, metric):

    for name, style in (
        ("sac_teacher", "--"),
        ("ppo_teacher", ":"),
    ):

        row = teacher[
            teacher.variant.eq(name)
        ]

        if len(row):

            ax.axhline(
                row.iloc[0][metric],
                color=COLORS[name],
                ls=style,
                lw=1.6,
                alpha=0.85,
                label=LABELS[name],
            )


# ---------------------------------------------------------
# 3. Compare BC training seeds for one dataset size
# ---------------------------------------------------------

def seed_scatter(bc, teacher, size, metric, output):

    fig, ax = plt.subplots(
        figsize=(8.8, 5.1)
    )

    group = bc[
        bc.dataset_size.eq(size)
    ]

    for j, variant in enumerate(VARIANTS):

        sub = group[group.variant.eq(variant)
        ].sort_values("training_seed")

        if sub.empty:
            continue

        vals = sub[metric].to_numpy(
            dtype=float
        )

        offsets = (
            np.linspace(-0.12, 0.12, len(sub))
            if len(sub) > 1
            else [0.0]
        )

        ax.scatter(
            j + np.asarray(offsets),
            vals,
            color=COLORS[variant],
            marker=MARKERS[variant],
            s=72,
            alpha=0.9,
            zorder=3,
            label=f"{LABELS[variant]} (seed)",
        )

        # Label individual BC training seeds.
        for x, row in zip(
            j + np.asarray(offsets),
            sub.itertuples()
        ):

            ax.annotate(
                str(int(row.training_seed)),
                (x, getattr(row, metric)),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # Mean across independently trained BC models.
        ax.hlines(
            vals.mean(),
            j - 0.25,
            j + 0.25,
            color=COLORS[variant],
            lw=3,
        )

    teacher_reference(
        ax,
        teacher,
        metric,
    )

    ax.set_xticks(
        range(len(VARIANTS)),
        [LABELS[v] for v in VARIANTS],
    )

    ax.set_ylabel(
        METRICS[metric][0]
    )

    ax.set_title(
        f"{size} demonstrations per dataset — "
        f"{metric.replace('_', ' ')}"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=170,
    )

    plt.close(fig)


# ---------------------------------------------------------
# 4. Compare dataset sizes (150 vs 300)
# ---------------------------------------------------------

def size_trend(bc, teacher, metric, output):

    fig, ax = plt.subplots(
        figsize=(9, 5.3)
    )

    sizes = sorted(
        bc.dataset_size.unique()
    )

    for variant in VARIANTS:

        sub = bc[
            bc.variant.eq(variant)
        ]

        if sub.empty:
            continue

        means = (
            sub.groupby("dataset_size")[metric]
            .mean()
            .sort_index()
        )

        # Mean performance across BC training seeds.
        ax.plot(
            means.index,
            means.values,
            color=COLORS[variant],
            marker=MARKERS[variant],
            lw=2,
            markersize=8,
            label=f"{LABELS[variant]} mean",
            zorder=4,
        )

        # Show individual training-seed results.
        for size in means.index:

            subsize = sub[
                sub.dataset_size.eq(size)
            ].sort_values("training_seed")

            offsets = (
                np.linspace(-3, 3, len(subsize))
                if len(subsize) > 1
                else [0]
            )

            ax.scatter(
                np.asarray([size] * len(subsize)) + offsets,
                subsize[metric],
                color=COLORS[variant],
                alpha=0.33,
                s=40,
                zorder=2,
            )

    teacher_reference(
        ax,
        teacher,
        metric,
    )

    ax.set_xticks(sizes)

    ax.set_xlabel(
        "Number of demonstration episodes in each BC dataset"
    )

    ax.set_ylabel(
        METRICS[metric][0]
    )

    ax.set_title(
        "Effect of demonstration quantity — "
        f"{metric.replace('_', ' ')}"
    )

    ax.grid(alpha=0.25)

    ax.legend(
        fontsize=8,
        ncol=2,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=170,
    )

    plt.close(fig)

# ---------------------------------------------------------
# 6. Performance gap relative to teacher
# ---------------------------------------------------------

def teacher_gap(bc, teacher, metric, output):
    """

    BC-High   -> SAC teacher
    BC-Low    -> PPO teacher

    Gap = BC metric - teacher metric

    """

    sources = {
        "high_quality": "sac_teacher",
        "low_quality": "ppo_teacher",
    }

    if not all(
        (teacher.variant == v).any()
        for v in sources.values()
    ):
        return False

    fig, ax = plt.subplots(
        figsize=(8.8, 4.8)
    )

    sizes = sorted(
        bc.dataset_size.unique()
    )

    for variant, teacher_name in sources.items():

        ref = float(teacher.loc[
                teacher.variant.eq(teacher_name),
                metric,
            ].iloc[0]
        )

        sub = bc[
            bc.variant.eq(variant)
        ].copy()

        sub["gap"] = sub[metric] - ref

        means = (
            sub.groupby("dataset_size")
            .gap.mean()
            .sort_index()
        )

        ax.plot(
            means.index,
            means.values,
            color=COLORS[variant],
            marker=MARKERS[variant],
            lw=2,
            label=(
                f"{LABELS[variant]} minus "
                f"{LABELS[teacher_name]}"
            ),
        )

        # Individual BC training-seed gaps.
        for size in means.index:

            pts = sub[
                sub.dataset_size.eq(size)
            ].sort_values("training_seed")

            offsets = (
                np.linspace(-3, 3, len(pts))
                if len(pts) > 1
                else [0]
            )

            ax.scatter(
                np.asarray([size] * len(pts)) + offsets,
                pts.gap,
                color=COLORS[variant],
                s=35,
                alpha=0.35,
            )

    ax.axhline(
        0,
        color="black",
        lw=1,
    )

    ax.set_xticks(sizes)

    ax.set_xlabel(
        "Demonstrations per dataset"
    )

    ax.set_ylabel(
        f"BC − teacher: {metric.replace('_', ' ')}"
    )

    ax.set_title(
        "Gap to corresponding teacher "
        "(zero = matched teacher)"
    )

    ax.grid(alpha=0.25)

    ax.legend(fontsize=9)

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=170,
    )

    plt.close(fig)

    return True


# ---------------------------------------------------------
# 7. Main: Generate all figures
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/model_summary.csv"),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/plots"),
    )

    args = parser.parse_args()

    bc, teacher = read_summary(
        args.summary
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for metric in METRICS:

        # Compare dataset sizes.
        size_trend(
            bc,
            teacher,
            metric,
            args.out / f"size_vs_{metric}.png",
        )

        count += 1

        # Compare independent seeds at each dataset size.
        for size in sorted(
            bc.dataset_size.unique()
        ):

            seed_scatter(
                bc,
                teacher,
                int(size),
                metric,
                args.out / f"n{int(size)}_{metric}.png",
            )

            count += 1

        # Compare BC performance against teacher.
        if metric in (
            "episode_return",
            "mean_distance",
            # "final_distance",
            # "success_2cm",
        ):

            if teacher_gap(
                bc,
                teacher,
                metric,
                args.out / f"teacher_gap_{metric}.png",
            ):
                count += 1

    # Reaching accuracy vs control-effort trade-off.
    for size in sorted(
        bc.dataset_size.unique()
    ):

        # tradeoff(
        #     bc,
        #     teacher,
        #     int(size),
        #     args.out / f"quality_tradeoff_n{int(size)}.png",
        # )

        count += 1

    print(
        f"Saved {count} figures to {args.out}"
    )


if __name__ == "__main__":
    main()
