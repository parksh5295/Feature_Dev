#!/usr/bin/env python3
"""
Plot grouping ablation summaries from CSV (preferred) or parsed .log files.

Example:
  python plot_grouping_ablation.py --csv results/grouping_ablation_both_seed42_anom5_20260406_074720.csv
  python plot_grouping_ablation.py --log results/grouping_ablation_both_seed42_anom5_20260406_074720.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_VARIANT_LINE = re.compile(r"^\s{2}(?P<v>coarse|baseline|fine)\s+n_elevated=(?P<e>\d+)")
_SPREAD_LINE = re.compile(r"^\s+spread\(max-min\)=(?P<s>\d+)")
_DATASET_LINE = re.compile(r"^# Dataset:\s*(?P<ds>\S+)")
_SEED_LINE = re.compile(r"# seed=(?P<seed>\d+)")


def parse_grouping_ablation_log(path: Path) -> pd.DataFrame:
    """Parse COMPARISON-style blocks from grouping_ablation_experiment .log output."""
    text = path.read_text(encoding="utf-8", errors="replace")
    seed: Optional[int] = None
    m = _SEED_LINE.search(text)
    if m:
        seed = int(m.group("seed"))

    rows: List[Dict[str, Any]] = []
    dataset: Optional[str] = None
    row_index: Optional[int] = None
    sample_order = 0
    pending: Dict[str, int] = {}

    for line in text.splitlines():
        if line.startswith("# seed=") and seed is None:
            m = _SEED_LINE.match(line.strip())
            if m:
                seed = int(m.group("seed"))
        dm = _DATASET_LINE.match(line)
        if dm:
            dataset = dm.group("ds").strip()
            sample_order = 0
            continue

        if line.startswith("--- Row index "):
            m = re.match(r"--- Row index (\d+) ---", line.strip())
            if m:
                row_index = int(m.group(1))
                sample_order += 1
                pending = {}
            continue

        vm = _VARIANT_LINE.match(line)
        if vm and dataset and row_index is not None:
            pending[vm.group("v")] = int(vm.group("e"))
            continue

        sm = _SPREAD_LINE.match(line)
        if sm and dataset and row_index is not None and len(pending) >= 3:
            spread = int(sm.group("s"))
            for v, elev in pending.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "row_index": row_index,
                        "sample_order": sample_order,
                        "variant": v,
                        "n_elevated": elev,
                        "spread": spread,
                    }
                )
            pending = {}
            row_index = None

    if not rows:
        raise ValueError(f"No comparison rows parsed from {path}")
    return pd.DataFrame(rows)


def load_table(csv_path: Optional[Path], log_path: Optional[Path]) -> pd.DataFrame:
    if csv_path is not None:
        return pd.read_csv(csv_path)
    if log_path is not None:
        return parse_grouping_ablation_log(log_path)
    raise ValueError("Provide --csv or --log")


def plot_grouped_bars(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    """Grouped bars: x = sample_order, hue = variant, facet by dataset."""
    df = df.copy()
    order_var = ["coarse", "baseline", "fine"]
    df["variant"] = pd.Categorical(df["variant"], categories=order_var, ordered=True)

    datasets = df["dataset"].unique().tolist()
    ncols = min(2, len(datasets))
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for ax, ds in zip(axes.flat, datasets):
        sub = df[df["dataset"] == ds]
        samples = sorted(sub["sample_order"].unique())
        x = np.arange(len(samples))
        width = 0.25
        for i, var in enumerate(order_var):
            heights = [
                float(sub[(sub["sample_order"] == s) & (sub["variant"] == var)]["n_elevated"].iloc[0])
                if len(sub[(sub["sample_order"] == s) & (sub["variant"] == var)]) > 0
                else 0.0
                for s in samples
            ]
            ax.bar(x + (i - 1) * width, heights, width, label=var)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in samples])
        ax.set_xlabel("sample_order (ablation row #)")
        ax.set_ylabel("n_elevated")
        ax.set_title(ds)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(datasets), nrows * ncols):
        axes.flat[j].set_visible(False)
    fig.suptitle(title or "Grouping ablation: elevated behavior groups per sample", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spread_scatter(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    """One point per (dataset, sample_order): spread value."""
    one = df.drop_duplicates(subset=["dataset", "sample_order", "spread"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for ds in one["dataset"].unique():
        sub = one[one["dataset"] == ds]
        ax.scatter(sub["sample_order"], sub["spread"], label=ds, s=60)
    ax.set_xlabel("sample_order")
    ax.set_ylabel("spread (max n_elevated - min across variants)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(title or "Sensitivity: spread per anomaly sample")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot grouping ablation CSV or log")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV from grouping_ablation_experiment.py")
    src.add_argument("--log", type=Path, help="grouping_ablation .log (parsed)")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: same dir as input")
    p.add_argument("--prefix", type=str, default="figure", help="Output filename prefix")
    args = p.parse_args(argv)

    inp = args.csv or args.log
    assert inp is not None
    if not inp.is_file():
        print(f"Not found: {inp}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp.stem

    df = load_table(args.csv, args.log)
    base = out_dir / f"{args.prefix}_{stem}"

    plot_grouped_bars(df, Path(str(base) + "_n_elevated.png"))
    plot_spread_scatter(df, Path(str(base) + "_spread.png"))

    print(f"Wrote {base}_n_elevated.png", file=sys.stderr)
    print(f"Wrote {base}_spread.png", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
