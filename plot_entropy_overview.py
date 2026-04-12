#!/usr/bin/env python3
"""
Paper-style overview: feature-level vs behavior-level entropy on anomaly rows.

For each anomaly, deviations |x_ij - mu_j| define a nonnegative vector; normalize to
probabilities and take Shannon H. Same for behavior scores D_ik (mean |x-mu| within groups).
Compare distributions with side-by-side boxplots and ΔH = H_feat − H_beh histograms.

Two datasets: 2×2 layout (NSL / NetML). One dataset: single row, two columns.

Example:
  python plot_entropy_overview.py --datasets both --out results/fig_entropy_overview.png
  python plot_entropy_overview.py --datasets nsl_kdd --anomaly-limit 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import behavior_deviation_experiment as bde
from plot_behavior_explanation import (
    FIG_FONT_PT,
    FIG_SAVE_DPI,
    _behavior_expl_rc,
    _netml_baseline_groups,
    _prepare_state,
)
from utils.nsl_kdd_config import NSL_KDD_BEHAVIOR_GROUPS, NSL_NORMAL_LABELS

_ROOT = Path(__file__).resolve().parent
_DEFAULT_NSL = _ROOT / "Dataset" / "KDDTest.csv"
_DEFAULT_NETML = _ROOT / "Dataset" / "netML_dataset.csv"
_RESULTS = _ROOT / "results"


def _sample_anomaly_indices(
    df_index: pd.Index,
    normal_mask: pd.Series,
    rng: np.random.Generator,
    limit: int,
    use_all: bool,
) -> np.ndarray:
    anomaly_idx = df_index[~normal_mask].to_numpy()
    if len(anomaly_idx) == 0:
        return anomaly_idx
    if use_all or len(anomaly_idx) <= limit:
        return anomaly_idx
    return rng.choice(anomaly_idx, size=limit, replace=False)


def _entropies_for_anomalies(
    X_all: pd.DataFrame,
    mu: pd.Series,
    groups: Mapping[str, List[str]],
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h_feat: List[float] = []
    h_beh: List[float] = []
    for idx in indices:
        row = X_all.loc[idx]
        dev = bde.feature_deviations_row(row, mu)
        v_f = dev.to_numpy(dtype=float)
        h_feat.append(bde.entropy_of_nonnegative_weights(v_f))
        scores = bde.behavior_scores_from_deviations(dev, groups)
        v_b = np.array(list(scores.values()), dtype=float)
        h_beh.append(bde.entropy_of_nonnegative_weights(v_b))
    return np.asarray(h_feat, dtype=float), np.asarray(h_beh, dtype=float)


def _run_one_dataset(
    name: str,
    df0: pd.DataFrame,
    groups: Mapping[str, List[str]],
    normal_pred: Callable[[str], bool],
    rng: np.random.Generator,
    anomaly_limit: int,
    all_anomalies: bool,
) -> Tuple[str, np.ndarray, np.ndarray]:
    X_all, df_w, mu, _q_hi, _q_vhi, normal_mask = _prepare_state(df0, groups, normal_pred)
    chosen = _sample_anomaly_indices(df_w.index, normal_mask, rng, anomaly_limit, all_anomalies)
    if len(chosen) == 0:
        return name, np.array([]), np.array([])
    hf, hb = _entropies_for_anomalies(X_all, mu, groups, chosen)
    return name, hf, hb


def plot_entropy_overview(
    series: List[Tuple[str, np.ndarray, np.ndarray]],
    out_path: Path,
    title: str = "",
) -> None:
    """series: list of (dataset_name, H_feat, H_beh)."""
    series = [(n, hf, hb) for n, hf, hb in series if len(hf) > 0 and len(hb) > 0]
    if not series:
        raise ValueError("No anomaly rows with valid entropies.")

    n_ds = len(series)
    with plt.rc_context(_behavior_expl_rc()):
        if n_ds == 1:
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
            grid = [(0, axes[0], axes[1], series[0])]
        else:
            fig, axes = plt.subplots(n_ds, 2, figsize=(11.0, 4.9 * n_ds))
            grid = [(i, axes[i, 0], axes[i, 1], series[i]) for i in range(n_ds)]

        xlabs = ["Feature-level\nentropy", "Behavior-level\nentropy"]
        for _, ax_box, ax_hist, (ds_name, hf, hb) in grid:
            data = [hf, hb]
            try:
                bp = ax_box.boxplot(
                    data,
                    tick_labels=xlabs,
                    patch_artist=True,
                    widths=0.55,
                )
            except TypeError:
                bp = ax_box.boxplot(
                    data,
                    labels=xlabs,
                    patch_artist=True,
                    widths=0.55,
                )
            for patch, c in zip(bp["boxes"], ("steelblue", "darkseagreen")):
                patch.set(facecolor=c, alpha=0.75)
            ax_box.set_ylabel(r"Entropy $H$ (nats)")
            ax_box.grid(axis="y", alpha=0.3)
            ax_box.set_title(f"{ds_name}: $H_{{\mathrm{{feat}}}}$ vs $H_{{\mathrm{{beh}}}}$ ($n={len(hf)}$)")

            delta = hf - hb
            ax_hist.hist(
                delta,
                bins=min(40, max(10, int(np.sqrt(len(delta))))),
                color="mediumpurple",
                edgecolor="white",
                alpha=0.88,
            )
            ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
            ax_hist.set_xlabel(r"$\Delta H = H_{\mathrm{feat}} - H_{\mathrm{beh}}$ (per row)")
            ax_hist.set_ylabel("Count")
            ax_hist.grid(axis="y", alpha=0.3)
            pos_share = 100.0 * float(np.mean(delta > 0)) if len(delta) else 0.0
            ax_hist.set_title(
                rf"{ds_name}: $\Delta H$ ($\Delta H>0$: {pos_share:.1f}\% of rows)"
            )

        fig.suptitle(
            title or "Explanation concentration: feature- vs behavior-level entropy",
            fontsize=FIG_FONT_PT + 1,
            y=1.02,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Feature vs behavior entropy overview (anomaly rows)")
    p.add_argument(
        "--datasets",
        choices=("nsl_kdd", "netml", "both"),
        default="both",
        help="Which dataset(s) to plot (both → 2×2 panels)",
    )
    p.add_argument("--nsl-path", type=Path, default=_DEFAULT_NSL)
    p.add_argument("--netml-path", type=Path, default=_DEFAULT_NETML)
    p.add_argument("--netml-label", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--anomaly-limit",
        type=int,
        default=8000,
        help="Max anomaly rows per dataset (subsample if larger; ignored with --all-anomalies)",
    )
    p.add_argument(
        "--all-anomalies",
        action="store_true",
        help="Use every anomaly row (can be slow on full NSL test)",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", type=str, default="")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out = args.out or (_RESULTS / "fig_entropy_overview.png")

    series: List[Tuple[str, np.ndarray, np.ndarray]] = []

    if args.datasets in ("nsl_kdd", "both"):
        if not args.nsl_path.is_file():
            print(f"NSL path not found: {args.nsl_path}", file=sys.stderr)
            if args.datasets == "nsl_kdd":
                return 1
        else:
            df_nsl = bde.load_nsl_kdd(args.nsl_path)
            groups = dict(NSL_KDD_BEHAVIOR_GROUPS)

            def nsl_pred(lab: str) -> bool:
                return bde._normalize_label(lab) in NSL_NORMAL_LABELS

            series.append(
                _run_one_dataset(
                    "NSL-KDD",
                    df_nsl,
                    groups,
                    nsl_pred,
                    rng,
                    args.anomaly_limit,
                    args.all_anomalies,
                )
            )

    if args.datasets in ("netml", "both"):
        if not args.netml_path.is_file():
            print(f"NetML path not found: {args.netml_path}", file=sys.stderr)
            if args.datasets == "netml":
                return 1
        else:
            df_nm = bde.load_netml_csv(args.netml_path, args.netml_label)
            groups_nm = _netml_baseline_groups(df_nm)
            pred_nm = bde.default_netml_normal_predicate()
            series.append(
                _run_one_dataset(
                    "NetML",
                    df_nm,
                    groups_nm,
                    pred_nm,
                    rng,
                    args.anomaly_limit,
                    args.all_anomalies,
                )
            )

    series = [s for s in series if len(s[1]) > 0]
    if not series:
        print("No data to plot (missing files or no anomalies).", file=sys.stderr)
        return 1

    try:
        plot_entropy_overview(series, out, title=args.title.strip())
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"Wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
