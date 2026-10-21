"""E1.1 Step 4: train the NumPy MLP and select hyperparameters on validation data.

Usage (from the repository root, with the virtual environment active):

    python experiments/E1.1-numpy-mlp-digits/run.py

Writes:
    experiments/E1.1-numpy-mlp-digits/RESULTS.md        human-readable summary
    experiments/E1.1-numpy-mlp-digits/metrics.json      machine-readable summary
    experiments/E1.1-numpy-mlp-digits/figures/*.png     learning curves
    runs/E1.1-numpy-mlp-digits/histories.json           every run's per-epoch metrics (git-ignored)

The test split is never evaluated here. It is reserved for the final comparison
against the baselines, so that no decision in this step can be tuned to it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from adaptive_neural_systems.data import load_digits_splits  # noqa: E402
from adaptive_neural_systems.training import TrainConfig, train  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGURES = HERE / "figures"
RUNS = ROOT / "runs" / HERE.name


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {p: version(p) for p in ("numpy", "scikit-learn", "matplotlib")},
    }


def mean_std(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def fmt(values, digits: int = 3) -> str:
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def main() -> None:
    cfg = json.loads((HERE / "config.json").read_text())
    search, data_cfg = cfg["search"], cfg["data"]
    s = load_digits_splits(
        data_cfg["split_seed"], data_cfg["val_fraction"], data_cfg["test_fraction"]
    )
    X_train, y_train, X_val, y_val = s.X_train, s.y_train, s.X_val, s.y_val
    # s.X_test / s.y_test are deliberately not used in this step.

    started = time.perf_counter()
    git = git_state()

    # --- 1. Small-batch overfitting check ------------------------------------------
    oc = cfg["overfit_check"]
    Xo, yo = X_train[: oc["n_examples"]], y_train[: oc["n_examples"]]
    _, overfit = train(
        Xo, yo, Xo, yo,
        TrainConfig(oc["learning_rate"], oc["epochs"], oc["n_examples"], oc["seed"]),
    )  # fmt: skip

    # --- 2. Grid search: every (lr, batch size) with every seed --------------------
    runs = []
    for lr, bs in product(search["learning_rates"], search["batch_sizes"]):
        histories = []
        for seed in search["seeds"]:
            _, h = train(
                X_train, y_train, X_val, y_val, TrainConfig(lr, search["epochs"], bs, seed)
            )
            histories.append(h)
        mean_val_loss = np.mean([h.val_loss for h in histories], axis=0)  # (epochs + 1,)
        best_epoch = int(np.argmin(mean_val_loss))
        runs.append(
            {
                "learning_rate": lr,
                "batch_size": bs,
                "histories": histories,
                "best_epoch": best_epoch,
                "best_mean_val_loss": float(mean_val_loss[best_epoch]),
            }
        )
        print(
            f"lr={lr:<5} bs={bs:<4} best mean val loss {mean_val_loss[best_epoch]:.4f} "
            f"at epoch {best_epoch}"
        )

    # --- 3. Select on validation loss ----------------------------------------------
    best = min(runs, key=lambda r: r["best_mean_val_loss"])
    e, hs = best["best_epoch"], best["histories"]
    at = {
        k: [getattr(h, k)[e] for h in hs]
        for k in ("train_loss", "val_loss", "train_acc", "val_acc")
    }
    final = {
        k: [getattr(h, k)[-1] for h in hs]
        for k in ("train_loss", "val_loss", "train_acc", "val_acc")
    }
    elapsed = time.perf_counter() - started

    # --- 4. Figures ------------------------------------------------------------------
    FIGURES.mkdir(exist_ok=True)
    epochs = np.arange(search["epochs"] + 1)
    colors = plt.cm.tab10.colors

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for i, (seed, h) in enumerate(zip(search["seeds"], hs, strict=True)):
        ax_loss.plot(
            epochs, h.train_loss, color=colors[i], ls="--", lw=1, label=f"train, seed {seed}"
        )
        ax_loss.plot(epochs, h.val_loss, color=colors[i], lw=1.5, label=f"validation, seed {seed}")
        ax_acc.plot(epochs, h.train_acc, color=colors[i], ls="--", lw=1)
        ax_acc.plot(epochs, h.val_acc, color=colors[i], lw=1.5)
    for ax in (ax_loss, ax_acc):
        ax.axvline(e, color="grey", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("cross-entropy (log scale)")
    ax_loss.legend(fontsize=7)
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.8, 1.005)
    fig.suptitle(
        f"Selected config: lr={best['learning_rate']}, batch={best['batch_size']} "
        f"(dotted line = selected epoch {e}); dashed = train, solid = validation"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(search["batch_sizes"]), figsize=(14, 4), sharey=True)
    for ax, bs in zip(axes, search["batch_sizes"], strict=True):
        for i, r in enumerate(x for x in runs if x["batch_size"] == bs):
            ax.plot(
                epochs,
                np.mean([h.val_loss for h in r["histories"]], axis=0),
                color=colors[i],
                label=f"lr={r['learning_rate']}",
            )
        ax.set_title(f"batch size {bs}")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean validation loss over seeds (log)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_loss_by_learning_rate.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(oc["epochs"] + 1), overfit.train_loss)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log)")
    ax.set_title(f"Overfitting check: {oc['n_examples']} examples, full batch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "overfit_check.png", dpi=120)
    plt.close(fig)

    # --- 5. Records ------------------------------------------------------------------
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "histories.json").write_text(
        json.dumps(
            [
                {
                    "learning_rate": r["learning_rate"],
                    "batch_size": r["batch_size"],
                    "seeds": search["seeds"],
                    "histories": [h.to_dict() for h in r["histories"]],
                }
                for r in runs
            ]
        )
    )

    metrics = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git,
        "environment": environment(),
        "wall_time_seconds": round(elapsed, 1),
        "config": cfg,
        "overfit_check": {
            "initial_loss": overfit.train_loss[0],
            "final_loss": overfit.train_loss[-1],
            "final_accuracy": overfit.train_acc[-1],
        },
        "grid": [
            {k: r[k] for k in ("learning_rate", "batch_size", "best_epoch", "best_mean_val_loss")}
            for r in runs
        ],
        "selected": {
            "learning_rate": best["learning_rate"],
            "batch_size": best["batch_size"],
            "epoch": e,
            "per_seed_at_selected_epoch": at,
            "per_seed_at_final_epoch": final,
        },
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    grid_rows = "\n".join(
        f"| {r['learning_rate']} | {r['batch_size']} | {r['best_epoch']} | "
        f"{r['best_mean_val_loss']:.4f} | "
        f"{fmt([h.val_acc[r['best_epoch']] for h in r['histories']])} |"
        for r in sorted(runs, key=lambda r: r["best_mean_val_loss"])
    )
    seed_rows = "\n".join(
        f"| {seed} | {at['train_loss'][i]:.4f} | {at['val_loss'][i]:.4f} | "
        f"{at['train_acc'][i]:.4f} | {at['val_acc'][i]:.4f} |"
        for i, seed in enumerate(search["seeds"])
    )
    results = f"""# E1.1 Step 4 results: training the NumPy MLP

*Generated by `run.py` on {metrics["generated_utc"]}. Do not edit by hand.*

- Git commit: `{git["commit"]}`{" (working tree had uncommitted changes)" if git["dirty"] else ""}
- Python {metrics["environment"]["python"]}, numpy {metrics["environment"]["packages"]["numpy"]}, \
scikit-learn {metrics["environment"]["packages"]["scikit-learn"]}, \
matplotlib {metrics["environment"]["packages"]["matplotlib"]}
- Platform: {metrics["environment"]["platform"]}; total wall time {elapsed:.1f} s
- Data: digits, split seed {data_cfg["split_seed"]}: {len(y_train)} train / \
{len(y_val)} validation / {len(s.y_test)} test (**test set not evaluated in this step**)

## Overfitting check ({oc["n_examples"]} training examples, full batch, lr {oc["learning_rate"]})

Training loss {overfit.train_loss[0]:.4f} → {overfit.train_loss[-1]:.4f} after {oc["epochs"]} \
epochs; training accuracy {overfit.train_acc[-1]:.4f}.

## Hyperparameter search (selection on validation loss only)

Grid: learning rates {search["learning_rates"]} × batch sizes {search["batch_sizes"]}, \
{search["epochs"]} epochs, seeds {search["seeds"]}. For each configuration the epoch \
with the lowest validation loss averaged over seeds is reported \
(this is early stopping chosen on validation data).

| learning rate | batch size | best epoch | mean val loss | val accuracy (mean ± std) |
| --- | --- | --- | --- | --- |
{grid_rows}

## Selected configuration: lr {best["learning_rate"]}, batch size {best["batch_size"]}, \
{e} epochs

| seed | train loss | val loss | train accuracy | val accuracy |
| --- | --- | --- | --- | --- |
{seed_rows}
| **mean ± std** | {fmt(at["train_loss"], 4)} | {fmt(at["val_loss"], 4)} | \
{fmt(at["train_acc"], 4)} | {fmt(at["val_acc"], 4)} |

Same configuration trained for all {search["epochs"]} epochs (no early stopping): train loss \
{fmt(final["train_loss"], 4)}, val loss {fmt(final["val_loss"], 4)}, val accuracy \
{fmt(final["val_acc"], 4)}.

Validation numbers here are optimistically biased: they were also used to choose the \
configuration. An unbiased estimate requires the untouched test set.

## Figures

- `figures/learning_curves.png`: train and validation loss/accuracy per seed, selected config
- `figures/validation_loss_by_learning_rate.png`: mean validation loss for every configuration
- `figures/overfit_check.png`: small-batch memorisation
"""
    (HERE / "RESULTS.md").write_text(results, encoding="utf-8")
    print(f"\nSelected lr={best['learning_rate']} batch={best['batch_size']} epoch={e}")
    print(f"val acc at selected epoch: {fmt(at['val_acc'], 4)}; wall time {elapsed:.1f}s")


if __name__ == "__main__":
    main()
