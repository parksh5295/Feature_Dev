#!/usr/bin/env python3
"""
Visualize feature-level vs behavior-level deviation for the proposed method.

1) Dual (two files): (a) top-k feature deviations |x-mu|; (b) per-behavior D_ik with q90/q99 lines.
2) Heatmaps: rows=anomaly samples, cols=behavior groups (baseline), NSL-KDD | NetML.

Example:
  python plot_behavior_explanation.py --mode dual --dataset nsl_kdd --row-index 17423
  # writes ..._row{id}_label_{name}_k{topk}_a.png (stable name; no date if same settings)
  # nonnormal: ..._label_nonnormal_...; omit --row-index to pick another row than --dual-label-tag auto (same --seed).
  # heatmap: ..._heatmap_{mode}_seed{seed}_n{n}_nonnormal.png or ..._label_{attack}.png
  python plot_behavior_explanation.py --mode heatmap --seed 42 --anomaly-samples 5
  python plot_behavior_explanation.py --mode heatmap --heatmap-attack-label ipsweep --seed 42 --anomaly-samples 5
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

import behavior_deviation_experiment as bde
from utils.nsl_kdd_config import NSL_KDD_BEHAVIOR_GROUPS, NSL_NORMAL_LABELS

_ROOT = Path(__file__).resolve().parent
_DEFAULT_NSL = _ROOT / "Dataset" / "KDDTest.csv"
_DEFAULT_NETML = _ROOT / "Dataset" / "netML_dataset.csv"
_RESULTS = _ROOT / "results"

# With --dual-label-tag nonnormal and no --row-index, use a different RNG stream than "auto"
# so the sample row differs from default auto (same --seed) — mixed-attack illustration.
_DUAL_NONNORMAL_ROW_SEED_OFFSET = 9_871_293

# Single fontsize for all text in a figure (bar, heatmap, dual, entropy overview @ 200 dpi).
FIG_FONT_FAMILY = "Times New Roman"
FIG_FONT_PT = 24
FIG_SAVE_DPI = 200


def _behavior_expl_rc() -> dict:
    return {
        "font.family": "serif",
        "font.serif": [FIG_FONT_FAMILY, "DejaVu Serif", "Times New Roman"],
        "font.size": FIG_FONT_PT,
        "axes.titlesize": FIG_FONT_PT,
        "axes.labelsize": FIG_FONT_PT,
        "xtick.labelsize": FIG_FONT_PT,
        "ytick.labelsize": FIG_FONT_PT,
        "legend.fontsize": FIG_FONT_PT,
        "figure.titlesize": FIG_FONT_PT,
    }


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


def _safe_label_for_filename(lab: str, max_len: int = 64) -> str:
    s = str(bde._normalize_label(lab))
    for c in '<>:"/\\|?*\n\r\t':
        s = s.replace(c, "_")
    s = "_".join(s.split())
    if len(s) > max_len:
        s = s[:max_len]
    return s or "unknown"


def _path_data_suffix(path: Path, default_path: Path) -> str:
    """Short tag when CSV path differs from default (same settings, different file)."""
    try:
        if path.resolve() == default_path.resolve():
            return ""
    except OSError:
        pass
    h = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"_data{h}"


def _wrap_xtick_label(name: str, max_line: int = 14) -> str:
    """Up to two lines, rotation 0; break at space when possible."""
    s = str(name).strip()
    if len(s) <= max_line:
        return s
    cut = s.rfind(" ", 0, max_line + 1)
    if cut < 4:
        cut = s.find(" ", max_line)
    if cut < 0:
        return s[:max_line] + "\n" + s[max_line:]
    a, b = s[:cut].strip(), s[cut:].strip()
    if len(b) > max_line * 2:
        b = b[: max_line * 2 - 1] + "…"
    return a + "\n" + b


def _heatmap_paths_suffix(nsl: Path, netml: Path, default_nsl: Path, default_netml: Path) -> str:
    try:
        if nsl.resolve() == default_nsl.resolve() and netml.resolve() == default_netml.resolve():
            return ""
    except OSError:
        pass
    h = hashlib.md5(f"{nsl.resolve()}|{netml.resolve()}".encode("utf-8")).hexdigest()[:8]
    return f"_data{h}"


def plot_panel_a(
    out_path: Path,
    dev: pd.Series,
    top_k: int,
) -> None:
    with plt.rc_context(_behavior_expl_rc()):
        top = dev.sort_values(ascending=False).head(top_k)
        n = len(top)
        fig, ax = plt.subplots(figsize=(8.0, max(4.2, 0.64 * n)))
        y_pos = np.arange(n)
        ax.barh(y_pos, top.values, height=0.72, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_truncate(str(i)) for i in top.index])
        ax.invert_yaxis()
        ax.set_xlabel(r"Feature deviation $|x_{ij}-\mu_j|$")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


def plot_panel_b(
    out_path: Path,
    scores: Mapping[str, float],
    q_hi: pd.Series,
    q_vhi: pd.Series,
) -> None:
    with plt.rc_context(_behavior_expl_rc()):
        beh_names = sorted(scores.keys())
        n_b = len(beh_names)
        fig = plt.figure(figsize=(10.8, max(5.4, 0.78 * n_b)))
        gs = gridspec.GridSpec(n_b, 1, figure=fig, hspace=0.62)

        for i, bname in enumerate(beh_names):
            ax = fig.add_subplot(gs[i, 0])
            D = scores[bname]
            qh = float(q_hi.get(bname, np.nan))
            qv = float(q_vhi.get(bname, np.nan))
            xmax = max(D, qh if np.isfinite(qh) else 0, qv if np.isfinite(qv) else 0) * 1.15
            xmax = max(xmax, 1e-12)
            ax.barh([0], [D], height=0.5, color="darkseagreen")
            if np.isfinite(qh):
                ax.axvline(qh, color="darkorange", linestyle="--", linewidth=2.0)
            if np.isfinite(qv):
                ax.axvline(qv, color="crimson", linestyle="--", linewidth=2.0)
            ax.set_xlim(0, xmax)
            # NetML 등 큰 스케일에서 Matplotlib가 축 모서리에 "1e6"/"1e7" 오프셋을 띄우는 것을 끔.
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
            ax.set_yticks([])
            ax.set_ylabel(_truncate(bname, 34), rotation=0, ha="right", va="center")
            ax.grid(axis="x", alpha=0.25)
            if i == n_b - 1:
                ax.set_xlabel(
                    r"Behavior deviation score (same units as $|x-\mu|$ within $G_k$)",
                    labelpad=12,
                )

        h_q90 = plt.Line2D(
            [0], [0], color="darkorange", linestyle="--", linewidth=2.0, label=r"$q_k^{(0.90)}$ (normal)"
        )
        h_q99 = plt.Line2D(
            [0], [0], color="crimson", linestyle="--", linewidth=2.0, label=r"$q_k^{(0.99)}$ (normal)"
        )
        h_d = plt.Rectangle((0, 0), 1, 1, fc="darkseagreen", label=r"$D_{ik}$")
        # Figure coords: lower y = further down. Leave room above bbox for xlabel + bottom spine ticks.
        fig.legend(
            handles=[h_d, h_q90, h_q99],
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
        )

        fig.tight_layout(rect=[0, 0.22, 1, 0.99])
        fig.savefig(out_path, dpi=FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


def plot_dual_separate(
    out_a: Path,
    out_b: Path,
    X_all: pd.DataFrame,
    mu: pd.Series,
    q_hi: pd.Series,
    q_vhi: pd.Series,
    groups: Mapping[str, List[str]],
    row_index: int,
    top_k: int,
) -> None:
    if row_index not in X_all.index:
        raise ValueError(f"row_index {row_index} not in processed frame index.")

    row = X_all.loc[row_index]
    dev = bde.feature_deviations_row(row, mu)
    scores = bde.behavior_scores_from_deviations(dev, groups)

    plot_panel_a(out_a, dev, top_k)
    plot_panel_b(out_b, scores, q_hi, q_vhi)


def plot_heatmap_pair(
    out_path: Path,
    nsl_path: Path,
    netml_path: Path,
    seed: int,
    n_anomaly: int,
    mode: str,
    attack_label_filter: Optional[str] = None,
) -> None:
    """mode: 'score' (continuous D_ik) or 'level' (binary elevated).
    attack_label_filter: if set, only rows whose normalized label equals this (e.g. ipsweep)."""

    def one_dataset(path: Path, load_fn, groups_fn, pred) -> Tuple[pd.DataFrame, List, str]:
        df0 = load_fn(path)
        groups = groups_fn(df0)
        X_all, df_w, mu, q_hi, q_vhi, normal_mask = _prepare_state(df0, groups, pred)
        lab_norm = df_w["label"].map(bde._normalize_label)
        anomaly_mask = ~normal_mask
        title_note = ""
        if attack_label_filter is not None:
            tgt = bde._normalize_label(attack_label_filter)
            filtered = anomaly_mask & (lab_norm == tgt)
            anomaly_idx = df_w.index[filtered]
            if len(anomaly_idx) == 0:
                anomaly_idx = df_w.index[anomaly_mask]
                title_note = f" (all non-normal; no “{attack_label_filter}” rows)"
        else:
            anomaly_idx = df_w.index[anomaly_mask]
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
        return (
            pd.DataFrame(mat, index=[str(int(x)) for x in chosen], columns=beh_cols),
            list(chosen),
            title_note,
        )

    def nsl_load(p: Path):
        return bde.load_nsl_kdd(p)

    def netml_load(p: Path):
        return bde.load_netml_csv(p, None)

    def nsl_pred(lab: str) -> bool:
        return bde._normalize_label(lab) in NSL_NORMAL_LABELS

    df_nsl, _, note_nsl = one_dataset(nsl_path, nsl_load, lambda d: dict(NSL_KDD_BEHAVIOR_GROUPS), nsl_pred)
    df_nm, _, note_nm = one_dataset(netml_path, netml_load, _netml_baseline_groups, bde.default_netml_normal_predicate())

    h_each = max(3.8, 0.58 * n_anomaly + 0.35)
    with plt.rc_context(_behavior_expl_rc()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, h_each * 2 + 0.9))

        if mode == "score":
            vmax_nsl = float(np.nanmax(df_nsl.values)) if np.isfinite(df_nsl.values.max()) else 1e-12
            vmax_nm = float(np.nanmax(df_nm.values)) if np.isfinite(df_nm.values.max()) else 1e-12
            vmax_nsl = max(vmax_nsl, 1e-12)
            vmax_nm = max(vmax_nm, 1e-12)
            im1 = ax1.imshow(df_nsl.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax_nsl)
            im2 = ax2.imshow(df_nm.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax_nm)
            cb1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
            cb1.set_label(r"$D_{ik}$ (NSL scale)")
            cb1.ax.tick_params(labelsize=FIG_FONT_PT)
            cb2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
            cb2.set_label(r"$D_{ik}$ (NetML scale)")
            cb2.ax.tick_params(labelsize=FIG_FONT_PT)
        else:
            im1 = ax1.imshow(df_nsl.values, aspect="auto", cmap="OrRd", vmin=0, vmax=1)
            im2 = ax2.imshow(df_nm.values, aspect="auto", cmap="OrRd", vmin=0, vmax=1)
            cb1 = fig.colorbar(im1, ax=ax1, fraction=0.035, pad=0.02)
            cb1.set_label("1 = elevated (↑/↑↑)")
            cb1.ax.tick_params(labelsize=FIG_FONT_PT)
            cb2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
            cb2.set_label("1 = elevated (↑/↑↑)")
            cb2.ax.tick_params(labelsize=FIG_FONT_PT)

        for ax, df, title, note in (
            (ax1, df_nsl, "NSL-KDD", note_nsl),
            (ax2, df_nm, "NetML", note_nm),
        ):
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_xticklabels(
                [_wrap_xtick_label(c) for c in df.columns],
                rotation=0,
                ha="center",
            )
            ax.set_yticks(np.arange(df.shape[0]))
            ax.set_yticklabels(df.index)
            ax.set_ylabel("anomaly row index")
            ax.set_title(title + note)
            ax.tick_params(axis="x", pad=2)

        fig.tight_layout()
        fig.savefig(out_path, dpi=FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot feature vs behavior deviation figures")
    p.add_argument("--mode", choices=("dual", "heatmap", "both"), default="dual")
    p.add_argument("--dataset", choices=("nsl_kdd", "netml"), default="nsl_kdd", help="Used for dual panel only")
    p.add_argument("--nsl-path", type=Path, default=_DEFAULT_NSL)
    p.add_argument("--netml-path", type=Path, default=_DEFAULT_NETML)
    p.add_argument("--row-index", type=int, default=None, help="DataFrame index for dual panel; default: first random anomaly")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--dual-label-tag",
        choices=("auto", "nonnormal"),
        default="auto",
        help=(
            "auto: filename includes this row's attack label. "
            "nonnormal: filename uses label_nonnormal; if --row-index is omitted, row is drawn with a "
            "separate seed from 'auto' (same --seed) so the plot usually differs from the auto case. "
            "If you set --row-index, the figure is that row only — same pixels as auto for the same row."
        ),
    )
    p.add_argument("--anomaly-samples", type=int, default=5)
    p.add_argument("--heatmap-mode", choices=("score", "level"), default="score")
    p.add_argument(
        "--heatmap-attack-label",
        type=str,
        default=None,
        metavar="LABEL",
        help="Only sample anomaly rows with this label (e.g. ipsweep). Default: all non-normal rows.",
    )
    p.add_argument("--out-dir", type=Path, default=_RESULTS)
    p.add_argument("--prefix", type=str, default="behavior_expl")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

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
            if args.dual_label_tag == "nonnormal":
                print(
                    "dual: --row-index is set; plot content is this row only. "
                    "Filename says label_nonnormal but values match the row's true label.",
                    file=sys.stderr,
                )
        else:
            pick_seed = (
                args.seed + _DUAL_NONNORMAL_ROW_SEED_OFFSET
                if args.dual_label_tag == "nonnormal"
                else args.seed
            )
            rng = np.random.default_rng(pick_seed)
            rid = int(rng.choice(anomaly_idx.to_numpy(), size=1)[0])
            if args.dual_label_tag == "nonnormal":
                print(
                    f"dual nonnormal: picked row_index={rid} (true label={df_w.loc[rid, 'label']!r})",
                    file=sys.stderr,
                )

        lbl_raw = df_w.loc[rid, "label"]
        if args.dual_label_tag == "nonnormal":
            lab_seg = "nonnormal"
        else:
            lab_seg = _safe_label_for_filename(lbl_raw)
        if args.dataset == "nsl_kdd":
            src_tag = _path_data_suffix(args.nsl_path, _DEFAULT_NSL)
        else:
            src_tag = _path_data_suffix(args.netml_path, _DEFAULT_NETML)
        base = f"{args.prefix}_dual_{args.dataset}_row{rid}_label_{lab_seg}_k{args.top_k}{src_tag}"
        out_a = args.out_dir / f"{base}_a.png"
        out_b = args.out_dir / f"{base}_b.png"
        plot_dual_separate(out_a, out_b, X_all, mu, q_hi, q_vhi, groups, rid, args.top_k)
        print(f"Wrote {out_a}", file=sys.stderr)
        print(f"Wrote {out_b}", file=sys.stderr)

    if args.mode in ("heatmap", "both"):
        htag = _heatmap_paths_suffix(args.nsl_path, args.netml_path, _DEFAULT_NSL, _DEFAULT_NETML)
        if args.heatmap_attack_label:
            lab_tag = f"_label_{_safe_label_for_filename(args.heatmap_attack_label)}"
        else:
            lab_tag = "_nonnormal"
        out = args.out_dir / (
            f"{args.prefix}_heatmap_{args.heatmap_mode}_seed{args.seed}_n{args.anomaly_samples}{lab_tag}{htag}.png"
        )
        plot_heatmap_pair(
            out,
            args.nsl_path,
            args.netml_path,
            args.seed,
            args.anomaly_samples,
            args.heatmap_mode,
            attack_label_filter=args.heatmap_attack_label,
        )
        print(f"Wrote {out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
