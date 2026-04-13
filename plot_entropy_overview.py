#!/usr/bin/env python3
"""
Paper-style overview: feature-level vs behavior-level entropy on anomaly rows.

Behavior entropy is computed under three grouping maps (coarse / baseline / fine),
matching grouping_ablation_experiment.py. The left panel shows four boxplots:
Feature entropy (invariant) and behavior entropy for each grouping variant.
The right panel overlays ΔH = H_feat − H_beh for each variant.

Example:
  python plot_entropy_overview.py --datasets netml --out results/fig_entropy_overview_netml.png
  python plot_entropy_overview.py --datasets nsl_kdd --anomaly-limit 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from matplotlib.axes import Axes

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import behavior_deviation_experiment as bde
import grouping_ablation_experiment as gae
from plot_behavior_explanation import (
    FIG_FONT_PT,
    FIG_SAVE_DPI,
    _behavior_expl_rc,
)
from utils.nsl_kdd_config import NSL_KDD_BEHAVIOR_GROUPS, NSL_NORMAL_LABELS

# Larger than dual/heatmap (same script family); entropy figure has dense x labels.
_ENTROPY_FONT_PT = FIG_FONT_PT + 10


def _entropy_overview_rc() -> dict:
    rc = dict(_behavior_expl_rc())
    for k in (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ):
        rc[k] = _ENTROPY_FONT_PT
    return rc


_ROOT = Path(__file__).resolve().parent
_DEFAULT_NSL = _ROOT / "Dataset" / "KDDTest.csv"
_DEFAULT_NETML = _ROOT / "Dataset" / "netML_dataset.csv"
_RESULTS = _ROOT / "results"

_VARIANT_ORDER = ("coarse", "baseline", "fine")


def _prepare_entropy_frame(
    df: pd.DataFrame,
    group_variants: Dict[str, Dict[str, List[str]]],
    normal_pred: Callable[[str], bool],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feat_set = set()
    for gd in group_variants.values():
        for fs in gd.values():
            feat_set.update(fs)
    feature_cols = [c for c in bde.numeric_feature_columns(df) if c in feat_set]
    X_all, cols = bde.prepare_numeric_frame(df, sorted(feature_cols))
    df_work = df.loc[X_all.index].copy()
    X_all.columns = cols
    labels = df_work["label"].map(bde._normalize_label)
    normal_mask = labels.map(normal_pred)
    if not normal_mask.any():
        raise ValueError("No normal samples for mu.")
    X_normal = X_all.loc[normal_mask]
    mu = bde.compute_normal_mean(X_normal)
    return X_all, df_work, mu, normal_mask


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


def _entropies_all_variants(
    X_all: pd.DataFrame,
    mu: pd.Series,
    group_variants: Dict[str, Dict[str, List[str]]],
    indices: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    h_feat: List[float] = []
    h_beh: Dict[str, List[float]] = {v: [] for v in _VARIANT_ORDER}
    for idx in indices:
        row = X_all.loc[idx]
        dev = bde.feature_deviations_row(row, mu)
        v_f = dev.to_numpy(dtype=float)
        h_feat.append(bde.entropy_of_nonnegative_weights(v_f))
        for v in _VARIANT_ORDER:
            scores = bde.behavior_scores_from_deviations(dev, group_variants[v])
            v_b = np.array(list(scores.values()), dtype=float)
            h_beh[v].append(bde.entropy_of_nonnegative_weights(v_b))
    return np.asarray(h_feat, dtype=float), {v: np.asarray(h_beh[v], dtype=float) for v in _VARIANT_ORDER}


def _nsl_group_variants() -> Dict[str, Dict[str, List[str]]]:
    return {
        "coarse": gae._nsl_coarse_groups(),
        "baseline": dict(NSL_KDD_BEHAVIOR_GROUPS),
        "fine": gae._nsl_fine_groups(),
    }


def _netml_group_variants(df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    return {
        "coarse": gae._netml_coarse_groups(df),
        "baseline": gae._netml_baseline_groups(df),
        "fine": gae._netml_fine_groups(df),
    }


def _run_one_dataset(
    name: str,
    df0: pd.DataFrame,
    group_variants: Dict[str, Dict[str, List[str]]],
    normal_pred: Callable[[str], bool],
    rng: np.random.Generator,
    anomaly_limit: int,
    all_anomalies: bool,
) -> Tuple[str, np.ndarray, Dict[str, np.ndarray]]:
    X_all, df_w, mu, normal_mask = _prepare_entropy_frame(df0, group_variants, normal_pred)
    chosen = _sample_anomaly_indices(df_w.index, normal_mask, rng, anomaly_limit, all_anomalies)
    if len(chosen) == 0:
        return name, np.array([]), {v: np.array([]) for v in _VARIANT_ORDER}
    hf, hb = _entropies_all_variants(X_all, mu, group_variants, chosen)
    return name, hf, hb


def plot_entropy_overview(
    series: List[Tuple[str, np.ndarray, Dict[str, np.ndarray]]],
    out_path: Path,
    title: str = "",
) -> None:
    series = [(n, hf, hb) for n, hf, hb in series if len(hf) > 0]
    if not series:
        raise ValueError("No anomaly rows with valid entropies.")

    n_ds = len(series)
    # Wider figure; left column wider than right (width_ratios) for x tick labels.
    fig_w = 16.8
    # Extra bottom space so legend sits below x-axis labels.
    fig_h = 5.75 * n_ds + 0.72 + 0.30 * max(0, n_ds - 1)

    # Same colors for coarse / baseline / fine on left (boxes 2–4) and right (histograms).
    c_feat = "#4682b4"
    c_variant = {"coarse": "#2171b5", "baseline": "#238b45", "fine": "#cb181d"}
    box_colors = (c_feat, c_variant["coarse"], c_variant["baseline"], c_variant["fine"])

    xlabs_left = [
        "Feature",
        "Behavior\n(coarse)",
        "Behavior\n(baseline)",
        "Behavior\n(fine)",
    ]
    legend_handles = [
        Patch(
            facecolor=c_feat,
            alpha=0.78,
            edgecolor="0.25",
            linewidth=0.5,
            label=r"$H_{\mathrm{feat}}$",
        ),
        Patch(
            facecolor=c_variant["coarse"],
            alpha=0.78,
            edgecolor="0.25",
            linewidth=0.5,
            label="Coarse",
        ),
        Patch(
            facecolor=c_variant["baseline"],
            alpha=0.78,
            edgecolor="0.25",
            linewidth=0.5,
            label="Baseline",
        ),
        Patch(
            facecolor=c_variant["fine"],
            alpha=0.78,
            edgecolor="0.25",
            linewidth=0.5,
            label="Fine",
        ),
    ]
    legend_labels = [h.get_label() for h in legend_handles]

    with plt.rc_context(_entropy_overview_rc()):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(
            n_ds,
            2,
            figure=fig,
            height_ratios=[1] * n_ds,
            width_ratios=[1.72, 0.82],
            hspace=0.48,
            # Extra gap so histogram y-label ("Count") does not crowd the boxplot panel.
            wspace=0.30,
        )
        row_axes: List[Tuple[Axes, Axes]] = []

        for row, (_, hf, h_beh) in enumerate(series):
            ax_box = fig.add_subplot(gs[row, 0])
            ax_hist = fig.add_subplot(gs[row, 1])

            data = [hf, h_beh["coarse"], h_beh["baseline"], h_beh["fine"]]
            try:
                bp = ax_box.boxplot(
                    data,
                    tick_labels=xlabs_left,
                    patch_artist=True,
                    widths=0.5,
                )
            except TypeError:
                bp = ax_box.boxplot(
                    data,
                    labels=xlabs_left,
                    patch_artist=True,
                    widths=0.5,
                )
            for patch, c in zip(bp["boxes"], box_colors):
                patch.set(facecolor=c, alpha=0.78)
            ax_box.set_ylabel(r"Entropy $H$ (nats)")
            ax_box.grid(axis="y", alpha=0.3)
            plt.setp(ax_box.get_xticklabels(), ha="center")

            deltas = [hf - h_beh[v] for v in _VARIANT_ORDER]
            stacked = np.hstack(deltas) if len(hf) else np.array([0.0])
            lo, hi = float(np.min(stacked)), float(np.max(stacked))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = (lo - 0.5, hi + 0.5) if lo == hi else (0.0, 1.0)
            n_bins = min(45, max(12, int(np.sqrt(len(hf)))))
            bin_edges = np.linspace(lo, hi, n_bins + 1)
            for v in _VARIANT_ORDER:
                ax_hist.hist(
                    hf - h_beh[v],
                    bins=bin_edges,
                    color=c_variant[v],
                    alpha=0.45,
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
            ax_hist.set_xlabel(r"$\Delta H = H_{\mathrm{feat}} - H_{\mathrm{beh}}$")
            ax_hist.set_ylabel("Count")
            ax_hist.grid(axis="y", alpha=0.3)
            row_axes.append((ax_box, ax_hist))

        if title.strip():
            fig.suptitle(title.strip(), fontsize=_ENTROPY_FONT_PT + 1, y=1.01)
        bottom_margin = 0.11 + 0.07 * max(0, n_ds - 1)
        fig.tight_layout(rect=(0.0, bottom_margin, 0.82, 0.98))
        fig.canvas.draw()
        # loc='upper center': top of legend box at y_anchor — place well below axis + xlabels.
        legend_below_axes = 0.118
        for ax_l, ax_r in row_axes:
            pos_l = ax_l.get_position()
            pos_r = ax_r.get_position()
            xc = (pos_l.x0 + pos_r.x1) / 2
            y_anchor = min(pos_l.y0, pos_r.y0) - legend_below_axes
            leg = Legend(
                fig,
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(xc, y_anchor),
                bbox_transform=fig.transFigure,
                ncol=4,
                frameon=True,
                fancybox=False,
                edgecolor="0.5",
            )
            fig.add_artist(leg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=FIG_SAVE_DPI, bbox_inches="tight", pad_inches=0.24)
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Feature vs behavior entropy overview (anomaly rows)")
    p.add_argument(
        "--datasets",
        choices=("nsl_kdd", "netml", "both"),
        default="both",
        help="Which dataset(s) to plot (both → stacked rows)",
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

    series: List[Tuple[str, np.ndarray, Dict[str, np.ndarray]]] = []

    if args.datasets in ("nsl_kdd", "both"):
        if not args.nsl_path.is_file():
            print(f"NSL path not found: {args.nsl_path}", file=sys.stderr)
            if args.datasets == "nsl_kdd":
                return 1
        else:
            df_nsl = bde.load_nsl_kdd(args.nsl_path)
            gv = _nsl_group_variants()

            def nsl_pred(lab: str) -> bool:
                return bde._normalize_label(lab) in NSL_NORMAL_LABELS

            series.append(
                _run_one_dataset(
                    "NSL-KDD",
                    df_nsl,
                    gv,
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
            gv_nm = _netml_group_variants(df_nm)
            pred_nm = bde.default_netml_normal_predicate()
            series.append(
                _run_one_dataset(
                    "NetML",
                    df_nm,
                    gv_nm,
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
