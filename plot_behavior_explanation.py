#!/usr/bin/env python3
"""
Visualize feature-level vs behavior-level deviation for the proposed method.

1) Dual panel: top-k feature deviations |x-mu| vs per-behavior D_ik with q90/q99 lines.
2) Heatmaps: rows=anomaly samples, cols=behavior groups (baseline), NSL-KDD | NetML.

Example:
  python plot_behavior_explanation.py --mode dual --dataset nsl_kdd --row-index 17423
  python plot_behavior_explanation.py --mode heatmap --seed 42 --anomaly-samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import behavior_deviation_experiment as bde
from utils.nsl_kdd_config import NSL_KDD_BEHAVIOR_GROUPS, NSL_NORMAL_LABELS

_ROOT = Path(__file__).resolve().parent
_DEFAULT_NSL = _ROOT / "Dataset" / "KDDTest.csv"
_DEFAULT_NETML = _ROOT / "Dataset" / "netML_dataset.csv"
_RESULTS = _ROOT / "results"


def _netml_baseline_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    num_cols = bde.numeric_feature_columns(df)
    groups: Dict[str, List[str]] = {}
    for c in num_cols:
        g = bde._infer_netml_behavior(c)
        groups.setdefault(g, []).append(c)
    return {k: v for k, v in groups.items() if v}


def _prepare_state(
    df: pd.DataFrame,
    groups: Mapping[str, List[str]],
    normal_pred: Callable[[str], bool],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    feature_cols = bde.numeric_feature_columns(df)
    grouped_feats = set()
    for fs in groups.values():
        grouped_feats.update(fs)
    feature_cols = [c for c in feature_cols if c in grouped_feats]

    X_all, cols = bde.prepare_numeric_frame(df, feature_cols)
    df_work = df.loc[X_all.index].copy()
    X_all.columns = cols

    labels = df_work["label"].map(bde._normalize_label)
    normal_mask = labels.map(normal_pred)
    if not normal_mask.any():
        raise ValueError("No normal samples for mu.")

    X_normal = X_all.loc[normal_mask]
    mu = bde.compute_normal_mean(X_normal)

    normal_behavior_rows: List[Dict[str, float]] = []
    for _, row in X_normal.iterrows():
        dev = bde.feature_deviations_row(row, mu)
        normal_behavior_rows.append(bde.behavior_scores_from_deviations(dev, groups))
    normal_scores = pd.DataFrame(normal_behavior_rows).fillna(0.0)
    q_hi, q_vhi = bde.fit_behavior_thresholds(normal_scores)
    return X_all, df_work, mu, q_hi, q_vhi, normal_mask


def _truncate(s: str, n: int = 32) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 2] + "…"


def plot_dual_panel(
    out_path: Path,
    X_all: pd.DataFrame,
    df_work: pd.DataFrame,
    mu: pd.Series,
    q_hi: pd.Series,
    q_vhi: pd.Series,
    groups: Mapping[str, List[str]],
    row_index: int,
    top_k: int,
    title_prefix: str,
) -> None:
    if row_index not in X_all.index:
        raise ValueError(f"row_index {row_index} not in processed frame index.")

    row = X_all.loc[row_index]
    lbl = df_work.loc[row_index, "label"]
    dev = bde.feature_deviations_row(row, mu)
    scores = bde.behavior_scores_from_deviations(dev, groups)

    beh_names = sorted(scores.keys())
    n_b = len(beh_names)
    fig = plt.figure(figsize=(12, max(4.0, 0.55 * n_b)))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.0], wspace=0.35)
    axL = fig.add_subplot(gs[0, 0])
    gs_r = gridspec.GridSpecFromSubplotSpec(n_b, 1, subplot_spec=gs[0, 1], hspace=0.4)

    top = dev.sort_values(ascending=False).head(top_k)
    y_pos = np.arange(len(top))
    axL.barh(y_pos, top.values, color="steelblue")
    axL.set_yticks(y_pos)
    axL.set_yticklabels([_truncate(str(i)) for i in top.index])
    axL.invert_yaxis()
    axL.set_xlabel(r"Feature deviation $|x_{ij}-\mu_j|$")
    axL.set_title("(a) Top feature deviations")
    axL.grid(axis="x", alpha=0.3)

    for i, bname in enumerate(beh_names):
        ax = fig.add_subplot(gs_r[i, 0])
        D = scores[bname]
        qh = float(q_hi.get(bname, np.nan))
        qv = float(q_vhi.get(bname, np.nan))
        xmax = max(D, qh if np.isfinite(qh) else 0, qv if np.isfinite(qv) else 0) * 1.15
        xmax = max(xmax, 1e-12)
        ax.barh([0], [D], height=0.5, color="darkseagreen")
        if np.isfinite(qh):
            ax.axvline(qh, color="darkorange", linestyle="--", linewidth=1.5)
        if np.isfinite(qv):
            ax.axvline(qv, color="crimson", linestyle="--", linewidth=1.5)
        ax.set_xlim(0, xmax)
        ax.set_yticks([])
        ax.set_ylabel(_truncate(bname, 34), fontsize=8, rotation=0, ha="right", va="center")
        ax.grid(axis="x", alpha=0.25)
        if i == 0:
            ax.set_title("(b) Behavior scores " + r"$D_{ik}$" + " vs normal quantiles")
        if i == n_b - 1:
            ax.set_xlabel(r"Behavior deviation score (same units as $|x-\mu|$ within $G_k$)")

    h_q90 = plt.Line2D([0], [0], color="darkorange", linestyle="--", linewidth=1.5, label=r"$q_k^{(0.90)}$ (normal)")
    h_q99 = plt.Line2D([0], [0], color="crimson", linestyle="--", linewidth=1.5, label=r"$q_k^{(0.99)}$ (normal)")
    h_d = plt.Rectangle((0, 0), 1, 1, fc="darkseagreen", label=r"$D_{ik}$")
    fig.legend(handles=[h_d, h_q90, h_q99], loc="upper center", ncol=3, bbox_to_anchor=(0.55, 0.02), fontsize=8)

    fig.suptitle(f"{title_prefix} | label={lbl} | row_index={row_index}", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_pair(
    out_path: Path,
    nsl_path: Path,
    netml_path: Path,
    seed: int,
    n_anomaly: int,
    mode: str,
) -> None:
    """mode: 'score' (continuous D_ik) or 'level' (binary elevated)."""

    def one_dataset(path: Path, load_fn, groups_fn, pred) -> Tuple[pd.DataFrame, List]:
        df0 = load_fn(path)
        groups = groups_fn(df0)
        X_all, df_w, mu, q_hi, q_vhi, normal_mask = _prepare_state(df0, groups, pred)
        anomaly_idx = df_w.index[~normal_mask]
        rng = np.random.default_rng(seed)
        n = min(n_anomaly, len(anomaly_idx))
        chosen = rng.choice(anomaly_idx.to_numpy(), size=n, replace=False)
        beh_cols = sorted(groups.keys())
        mat = np.full((len(chosen), len(beh_cols)), np.nan)
        for i, idx in enumerate(chosen):
            row = X_all.loc[idx]
            dev = bde.feature_deviations_row(row, mu)
            sc = bde.behavior_scores_from_deviations(dev, groups)
            for j, b in enumerate(beh_cols):
                if b not in sc:
                    continue
                D = sc[b]
                if mode == "score":
                    mat[i, j] = D
                else:
                    lev = bde.label_behavior_level(D, float(q_hi.get(b, np.inf)), float(q_vhi.get(b, np.inf)))
                    mat[i, j] = 1.0 if lev in ("↑", "↑↑") else 0.0
        return pd.DataFrame(mat, index=[str(int(x)) for x in chosen], columns=beh_cols), list(chosen)

    def nsl_load(p: Path):
        return bde.load_nsl_kdd(p)

    def netml_load(p: Path):
        return bde.load_netml_csv(p, None)

    def nsl_pred(lab: str) -> bool:
        return bde._normalize_label(lab) in NSL_NORMAL_LABELS

    df_nsl, _ = one_dataset(nsl_path, nsl_load, lambda d: dict(NSL_KDD_BEHAVIOR_GROUPS), nsl_pred)
    df_nm, _ = one_dataset(netml_path, netml_load, _netml_baseline_groups, bde.default_netml_normal_predicate())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(3.5, 0.45 * n_anomaly)))

    if mode == "score":
        # Per-panel vmax: a single huge NetML cell must not wash out NSL-KDD colors.
        vmax_nsl = float(np.nanmax(df_nsl.values)) if np.isfinite(df_nsl.values.max()) else 1e-12
        vmax_nm = float(np.nanmax(df_nm.values)) if np.isfinite(df_nm.values.max()) else 1e-12
        vmax_nsl = max(vmax_nsl, 1e-12)
        vmax_nm = max(vmax_nm, 1e-12)
        im1 = ax1.imshow(df_nsl.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax_nsl)
        im2 = ax2.imshow(df_nm.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax_nm)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label=r"$D_{ik}$ (NSL scale)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label=r"$D_{ik}$ (NetML scale)")
    else:
        im1 = ax1.imshow(df_nsl.values, aspect="auto", cmap="OrRd", vmin=0, vmax=1)
        im2 = ax2.imshow(df_nm.values, aspect="auto", cmap="OrRd", vmin=0, vmax=1)
        fig.colorbar(im1, ax=[ax1, ax2], fraction=0.02, pad=0.04, label="1 = elevated (↑/↑↑)")

    for ax, df, title in ((ax1, df_nsl, "NSL-KDD"), (ax2, df_nm, "NetML")):
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels([_truncate(c, 18) for c in df.columns], rotation=55, ha="right", fontsize=7)
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(df.index, fontsize=8)
        ax.set_ylabel("anomaly row index")
        ax.set_title(title)

    fig.suptitle(
        f"Behavior-level scores ({mode}) | seed={seed}, n={n_anomaly}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot feature vs behavior deviation figures")
    p.add_argument("--mode", choices=("dual", "heatmap", "both"), default="both")
    p.add_argument("--dataset", choices=("nsl_kdd", "netml"), default="nsl_kdd", help="Used for dual panel only")
    p.add_argument("--nsl-path", type=Path, default=_DEFAULT_NSL)
    p.add_argument("--netml-path", type=Path, default=_DEFAULT_NETML)
    p.add_argument("--row-index", type=int, default=None, help="DataFrame index for dual panel; default: first random anomaly")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--anomaly-samples", type=int, default=5)
    p.add_argument("--heatmap-mode", choices=("score", "level"), default="score")
    p.add_argument("--out-dir", type=Path, default=_RESULTS)
    p.add_argument("--prefix", type=str, default="behavior_expl")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    if args.mode in ("dual", "both"):
        if args.dataset == "nsl_kdd":
            df0 = bde.load_nsl_kdd(args.nsl_path)
            groups = dict(NSL_KDD_BEHAVIOR_GROUPS)

            def pred(lab: str) -> bool:
                return bde._normalize_label(lab) in NSL_NORMAL_LABELS
        else:
            df0 = bde.load_netml_csv(args.netml_path, None)
            groups = _netml_baseline_groups(df0)
            pred = bde.default_netml_normal_predicate()

        X_all, df_w, mu, q_hi, q_vhi, normal_mask = _prepare_state(df0, groups, pred)
        anomaly_idx = df_w.index[~normal_mask]
        if len(anomaly_idx) == 0:
            print("No anomaly rows.", file=sys.stderr)
            return 1
        if args.row_index is not None:
            rid = args.row_index
            if rid not in X_all.index:
                print(f"row_index {rid} not in data.", file=sys.stderr)
                return 1
        else:
            rng = np.random.default_rng(args.seed)
            rid = int(rng.choice(anomaly_idx.to_numpy(), size=1)[0])

        out = args.out_dir / f"{args.prefix}_dual_{args.dataset}_row{rid}_{stamp}.png"
        plot_dual_panel(out, X_all, df_w, mu, q_hi, q_vhi, groups, rid, args.top_k, args.dataset)
        print(f"Wrote {out}", file=sys.stderr)

    if args.mode in ("heatmap", "both"):
        out = args.out_dir / f"{args.prefix}_heatmap_{args.heatmap_mode}_seed{args.seed}_n{args.anomaly_samples}_{stamp}.png"
        plot_heatmap_pair(out, args.nsl_path, args.netml_path, args.seed, args.anomaly_samples, args.heatmap_mode)
        print(f"Wrote {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
