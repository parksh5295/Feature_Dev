#!/usr/bin/env python3
"""
Plot grouping ablation summaries from CSV (preferred) or parsed .log files.

Supports:
  - sample scope: grouped bars + spread scatter (few rows).
  - scope=all / by_label: boxplot of n_elevated, spread histogram, mean bars,
    optional heatmap (attack_label × variant).

Example:
  python plot_grouping_ablation.py --csv results/grouping_ablation_both_seed42_anom5_20260406_074720.csv
  python plot_grouping_ablation.py --csv results/grouping_ablation_nsl_kdd_scope-all_seed42_....csv --style aggregate
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


_VARIANT_ORDER = ["coarse", "baseline", "fine"]


def _should_use_aggregate(df: pd.DataFrame) -> bool:
    if len(df) > 400:
        return True
    if "slice_key" in df.columns:
        sk = df["slice_key"].astype(str)
        if sk.eq("__all__").all():
            return True
    if "attack_label" in df.columns and df["attack_label"].nunique() > 3 and len(df) > 500:
        return True
    return False


def plot_aggregate_figures(df: pd.DataFrame, out_base: Path, heatmap_top: int = 32) -> List[Path]:
    """Figures for scope=all / by_label CSVs (many rows). Returns paths written."""
    df = df.copy()
    df["variant"] = pd.Categorical(df["variant"], categories=_VARIANT_ORDER, ordered=True)
    written: List[Path] = []
    has_ds = "dataset" in df.columns
    datasets = df["dataset"].unique().tolist() if has_ds else ["_"]

    for ds in datasets:
        sub = df[df["dataset"] == ds] if has_ds else df
        tag = f"{ds}_" if has_ds else ""

        # --- Boxplot: n_elevated by variant
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = [sub[sub["variant"] == v]["n_elevated"].dropna().to_numpy() for v in _VARIANT_ORDER]
        try:
            bp = ax.boxplot(data, tick_labels=_VARIANT_ORDER, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=_VARIANT_ORDER, patch_artist=True)
        for p in bp["boxes"]:
            p.set(facecolor="lightsteelblue", alpha=0.85)
        ax.set_ylabel("n_elevated")
        ax.set_xlabel("grouping variant")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"n_elevated distribution ({ds})" if has_ds else "n_elevated distribution")
        fig.tight_layout()
        pth = Path(str(out_base) + f"_{tag}agg_box.png")
        fig.savefig(pth, dpi=150)
        plt.close(fig)
        written.append(pth)

        # --- Histogram: spread (one value per anomaly row)
        sp = sub.drop_duplicates(subset=["row_index"], keep="first")
        spreads = sp["spread"].dropna().to_numpy()
        fig, ax = plt.subplots(figsize=(7, 4))
        if len(spreads) > 0:
            ax.hist(spreads, bins=min(40, max(10, int(np.sqrt(len(spreads))))), color="coral", edgecolor="white")
        ax.set_xlabel("spread (max n_elevated − min across variants)")
        ax.set_ylabel("count (anomaly rows)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Spread distribution ({ds})" if has_ds else "Spread distribution")
        fig.tight_layout()
        pth = Path(str(out_base) + f"_{tag}agg_spread_hist.png")
        fig.savefig(pth, dpi=150)
        plt.close(fig)
        written.append(pth)

        # --- Mean ± std bars per variant
        g = sub.groupby("variant", observed=True)["n_elevated"]
        means = g.mean().reindex(_VARIANT_ORDER)
        stds = g.std().reindex(_VARIANT_ORDER).fillna(0)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(_VARIANT_ORDER))
        ax.bar(x, means, yerr=stds, capsize=4, color="seagreen", alpha=0.85, ecolor="dimgray")
        ax.set_xticks(x)
        ax.set_xticklabels(_VARIANT_ORDER)
        ax.set_ylabel("mean n_elevated ± std")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"Mean n_elevated by grouping ({ds})" if has_ds else "Mean n_elevated by grouping")
        fig.tight_layout()
        pth = Path(str(out_base) + f"_{tag}agg_mean_bars.png")
        fig.savefig(pth, dpi=150)
        plt.close(fig)
        written.append(pth)

        # --- Heatmap: attack_label × variant (mean n_elevated)
        if "attack_label" not in sub.columns:
            continue
        n_att = sub["attack_label"].nunique()
        if n_att < 2:
            continue
        counts = (
            sub.groupby("attack_label", observed=True)["row_index"].nunique().sort_values(ascending=False)
        )
        top_idx = counts.head(heatmap_top).index
        sub_top = sub[sub["attack_label"].isin(top_idx)]
        pv = sub_top.pivot_table(
            index="attack_label",
            columns="variant",
            values="n_elevated",
            aggfunc="mean",
            observed=False,
        )
        pv = pv.reindex(columns=[c for c in _VARIANT_ORDER if c in pv.columns])
        pv = pv.reindex(index=[i for i in counts.index if i in pv.index])

        fig, ax = plt.subplots(figsize=(8, max(5, 0.22 * len(pv.index))))
        im = ax.imshow(pv.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(np.arange(len(pv.columns)))
        ax.set_xticklabels(list(pv.columns), rotation=0)
        ax.set_yticks(np.arange(len(pv.index)))
        ax.set_yticklabels([str(i)[:28] for i in pv.index], fontsize=7)
        ax.set_xlabel("grouping variant")
        ax.set_ylabel("attack_label (top by row count)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="mean n_elevated")
        fig.tight_layout()
        pth = Path(str(out_base) + f"_{tag}agg_attack_heatmap.png")
        fig.savefig(pth, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(pth)

    return written


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


def _can_sample_plot(df: pd.DataFrame) -> bool:
    if "sample_order" not in df.columns:
        return False
    if len(df) > 3000:
        return False
    return int(df["sample_order"].nunique()) <= 50


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot grouping ablation CSV or log")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV from grouping_ablation_experiment.py")
    src.add_argument("--log", type=Path, help="grouping_ablation .log (parsed)")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: same dir as input")
    p.add_argument("--prefix", type=str, default="fig", help="Output filename prefix")
    p.add_argument(
        "--style",
        choices=("auto", "aggregate", "sample", "both"),
        default="auto",
        help="aggregate: box/hist/mean/heatmap (large CSV); sample: per-sample bars",
    )
    p.add_argument("--heatmap-top", type=int, default=32, help="Top attack labels by count in heatmap")
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

    style = args.style
    if style == "auto":
        style = "aggregate" if _should_use_aggregate(df) else "sample"

    written: List[str] = []

    if style in ("aggregate", "both"):
        paths = plot_aggregate_figures(df, base, heatmap_top=args.heatmap_top)
        for pth in paths:
            written.append(str(pth))
            print(f"Wrote {pth}", file=sys.stderr)

    if style in ("sample", "both"):
        if not _can_sample_plot(df):
            if style == "sample":
                print(
                    "CSV is too large for sample-style plots; use --style aggregate",
                    file=sys.stderr,
                )
                return 1
        else:
            p1 = Path(str(base) + "_n_elevated.png")
            p2 = Path(str(base) + "_spread.png")
            plot_grouped_bars(df, p1)
            plot_spread_scatter(df, p2)
            written.extend([str(p1), str(p2)])
            print(f"Wrote {p1}", file=sys.stderr)
            print(f"Wrote {p2}", file=sys.stderr)

    if not written:
        print("No figures written.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
