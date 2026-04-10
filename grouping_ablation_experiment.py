#!/usr/bin/env python3
"""
Feature-grouping sensitivity (ablation): coarse / baseline / fine behavior maps.

Runs the same anomaly rows three times with different groupings, logs full outputs
plus a short comparison (elevated-behavior counts and side-by-side summaries).

--scope sample: random N anomalies (default; verbose per sample).
--scope all:    every non-normal row in one run (quiet; aggregate in log).
--scope by_label: one run per attack label (normalized), same as all within each label.
"""

from __future__ import annotations

import argparse
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import behavior_deviation_experiment as bde
from utils.nsl_kdd_config import NSL_KDD_BEHAVIOR_GROUPS as NSL_BASELINE_GROUPS
from utils.nsl_kdd_config import NSL_NORMAL_LABELS

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET_DIR = _ROOT / "Dataset"
# Test set default: ablation runs three grouping variants (each scans normal rows).
_DEFAULT_NSL_KDD_CSV = _DEFAULT_DATASET_DIR / "KDDTest.csv"
_DEFAULT_NETML_CSV = _DEFAULT_DATASET_DIR / "netML_dataset.csv"
_RESULTS_DIR = _ROOT / "results"

VariantName = str


def _nsl_coarse_groups() -> Dict[str, List[str]]:
    b = NSL_BASELINE_GROUPS
    return {
        "Volume and timing": b["Data volume"] + b["Timing pattern"],
        "Connection and protocol": b["Connection intensity"] + b["Protocol / error rates"],
        "Host access / shell": list(b["Host access / shell"]),
    }


def _nsl_fine_groups() -> Dict[str, List[str]]:
    b = NSL_BASELINE_GROUPS
    return {
        "Connection intensity": list(b["Connection intensity"]),
        "Byte volume": ["src_bytes", "dst_bytes"],
        "Packet-level signals": ["wrong_fragment", "urgent"],
        "Timing (duration)": ["duration"],
        "Error rates": [
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
        ],
        "Service / diversity rates": [
            "same_srv_rate",
            "diff_srv_rate",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
        ],
        "Host access / shell": list(b["Host access / shell"]),
    }


def _netml_baseline_groups(df) -> Dict[str, List[str]]:
    num_cols = bde.numeric_feature_columns(df)
    groups: Dict[str, List[str]] = {}
    for c in num_cols:
        g = bde._infer_netml_behavior(c)
        groups.setdefault(g, []).append(c)
    return {k: v for k, v in groups.items() if v}


def _netml_coarse_groups(df) -> Dict[str, List[str]]:
    g0 = _netml_baseline_groups(df)
    return {
        "Volume and timing": g0.get("Timing pattern", []) + g0.get("Data volume", []),
        "Flow and protocol": g0.get("Connection intensity", []) + g0.get("Protocol / flags", []),
        "Other features": g0.get("Other features", []),
    }


def _infer_netml_behavior_fine(column: str) -> str:
    base = bde._infer_netml_behavior(column)
    if base != "Timing pattern":
        return base
    c = column
    if re.search(r"(?i)\biat\b", c):
        return "IAT timing"
    if re.search(r"(?i)\b(idle|active)\b", c):
        return "Idle / active timing"
    if re.search(r"(?i)duration|flow_?dur", c):
        return "Flow duration"
    return "Other timing"


def _netml_fine_groups(df) -> Dict[str, List[str]]:
    num_cols = bde.numeric_feature_columns(df)
    groups: Dict[str, List[str]] = {}
    for c in num_cols:
        g = _infer_netml_behavior_fine(c)
        groups.setdefault(g, []).append(c)
    return {k: v for k, v in groups.items() if v}


def _pick_anomaly_indices(df: pd.DataFrame, normal_pred: Callable, limit: int, seed: int):
    labels = df["label"].map(bde._normalize_label)
    normal_mask = labels.map(normal_pred)
    anomaly_idx = df.index[~normal_mask]
    if len(anomaly_idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    n = min(limit, len(anomaly_idx))
    return rng.choice(anomaly_idx.to_numpy(), size=n, replace=False)


def _all_anomaly_indices(df: pd.DataFrame, normal_pred: Callable) -> np.ndarray:
    labels = df["label"].map(bde._normalize_label)
    normal_mask = labels.map(normal_pred)
    return df.index[~normal_mask].to_numpy()


def _capture_run(
    df,
    groups: Mapping[str, List[str]],
    normal_pred: Callable,
    chosen: np.ndarray,
    seed: int,
    quiet: bool = False,
) -> Tuple[str, List[Tuple[str, int]]]:
    """Returns stdout capture and list of (sample_key, n_elevated) per anomaly row."""
    if quiet:
        res = bde.run_experiment(
            df,
            groups,
            normal_pred,
            anomaly_sample_limit=len(chosen),
            random_state=seed,
            anomaly_indices=chosen,
            quiet=True,
        )
        assert res is not None
        summaries: List[Tuple[str, int]] = []
        for idx, elev in res:
            lbl = df.loc[idx, "label"]
            summaries.append((f"--- 샘플 | label={lbl} | idx={idx}", elev))
        return "", summaries

    buf = StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        bde.run_experiment(
            df,
            groups,
            normal_pred,
            anomaly_sample_limit=len(chosen),
            random_state=seed,
            anomaly_indices=chosen,
        )
    finally:
        sys.stdout = old
    text = buf.getvalue()
    summaries = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("--- 샘플 ") and "| label=" in lines[i]:
            sample_line = lines[i].strip()
            elevated = 0
            j = i + 1
            while j < len(lines) and not lines[j].startswith("--- 샘플 "):
                if "→ ↑" in lines[j] or "→ ↑↑" in lines[j]:
                    elevated += 1
                j += 1
            summaries.append((sample_line, elevated))
            i = j
        else:
            i += 1
    return text, summaries


def _write_per_label_aggregates(logf, rows: List[Dict[str, Any]]) -> None:
    """One row per (row_index, variant); writes mean n_elevated per attack label."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "attack_label" not in df.columns or "variant" not in df.columns:
        return
    logf.write("\n" + "=" * 72 + "\n")
    logf.write("# PER-LABEL AGGREGATES (single pass over all anomalies; grouped by attack_label)\n")
    logf.write("=" * 72 + "\n")
    vnames = sorted(df["variant"].unique())
    for lab in sorted(df["attack_label"].unique()):
        sub = df[df["attack_label"] == lab]
        n_anom = len(sub) // len(vnames) if vnames else 0
        logf.write(f"\n## attack_label={lab}  (n≈{n_anom} anomaly rows)\n")
        for vn in vnames:
            s = sub[sub["variant"] == vn]["n_elevated"].dropna()
            if len(s) == 0:
                continue
            arr = s.to_numpy(dtype=float)
            logf.write(
                f"  {vn:12s}  n={len(arr)}  mean={arr.mean():.6g}  std={arr.std():.6g}  median={np.median(arr):.6g}\n"
            )
        spreads = []
        for rid in sub["row_index"].unique():
            rsub = sub[sub["row_index"] == rid]
            vals: List[float] = []
            for vn in vnames:
                q = rsub[rsub["variant"] == vn]["n_elevated"]
                if len(q) == 0:
                    break
                vals.append(float(q.iloc[0]))
            if len(vals) == len(vnames):
                spreads.append(float(max(vals) - min(vals)))
        if spreads:
            sp = np.asarray(spreads, dtype=float)
            logf.write(
                f"  spread(max-min)  mean={sp.mean():.6g}  median={np.median(sp):.6g}\n"
            )


def _write_aggregate(
    logf,
    all_summaries: Dict[str, List[Tuple[str, int]]],
    vnames: List[str],
    n_rows: int,
) -> None:
    logf.write("\n# AGGREGATE (n_elevated)\n")
    for vn in vnames:
        s = all_summaries.get(vn, [])
        vals = [t[1] for t in s]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        logf.write(
            f"  {vn:12s}  n={len(vals)}  mean={arr.mean():.6g}  std={arr.std():.6g}  median={np.median(arr):.6g}\n"
        )
    spreads: List[float] = []
    for idx in range(n_rows):
        counts: List[int] = []
        ok = True
        for vn in vnames:
            s = all_summaries.get(vn, [])
            if idx >= len(s):
                ok = False
                break
            counts.append(s[idx][1])
        if ok and counts:
            spreads.append(float(max(counts) - min(counts)))
    if spreads:
        sp = np.asarray(spreads, dtype=float)
        logf.write(
            f"  spread(max-min)  mean={sp.mean():.6g}  median={np.median(sp):.6g}\n"
        )


def _run_dataset(
    name: str,
    df,
    variants: Dict[VariantName, Mapping[str, List[str]]],
    normal_pred: Callable,
    chosen: np.ndarray,
    seed: int,
    logf,
    slice_key: str,
    quiet: bool,
    slice_key_per_attack: bool = False,
) -> List[Dict[str, Any]]:
    logf.write(f"\n{'=' * 72}\n")
    logf.write(f"# Dataset: {name}\n")
    logf.write(f"# slice: {slice_key}\n")
    if slice_key_per_attack:
        logf.write("# slice_key per row = normalized attack label (by_label mode)\n")
    if quiet and len(chosen) > 20:
        logf.write(f"# anomaly rows: n={len(chosen)} (indices omitted in header)\n")
    else:
        logf.write(f"# Same anomaly rows (indices): {list(chosen)}\n")
    logf.write(f"{'=' * 72}\n")

    all_summaries: Dict[VariantName, List[Tuple[str, int]]] = {}
    for vname, groups in variants.items():
        n_groups = len(groups)
        n_feats = sum(len(fs) for fs in groups.values())
        logf.write(f"\n### Variant: {vname} ({n_groups} behavior groups, {n_feats} features mapped)\n")
        logf.write("### Group -> features:\n")
        for gname, feats in sorted(groups.items(), key=lambda x: x[0]):
            logf.write(f"  - {gname}: {len(feats)} features\n")
        logf.write("\n")

        block, summ = _capture_run(df, groups, normal_pred, chosen, seed, quiet=quiet)
        if quiet:
            logf.write("(quiet: per-sample stdout omitted)\n")
        else:
            logf.write(block)
        if not summ:
            logf.write("(no per-sample summary parsed)\n")
        all_summaries[vname] = summ

    logf.write(f"\n{'=' * 72}\n")
    logf.write("# COMPARISON (same rows across variants)\n")
    logf.write("# n_elevated = count of behavior groups with ↑ or ↑↑ for that sample.\n")
    logf.write(f"{'=' * 72}\n")
    vnames = list(variants.keys())
    if quiet:
        _write_aggregate(logf, all_summaries, vnames, len(chosen))
    else:
        for idx in range(len(chosen)):
            logf.write(f"\n--- Row index {chosen[idx]} ---\n")
            counts = []
            for vn in vnames:
                s = all_summaries.get(vn, [])
                if idx < len(s):
                    line, elev = s[idx]
                    logf.write(f"  {vn:12s}  n_elevated={elev}  |  {line}\n")
                    counts.append(elev)
                else:
                    logf.write(f"  {vn:12s}  (missing)\n")
            if len(counts) == len(vnames) and counts:
                spread = max(counts) - min(counts)
                logf.write(f"  spread(max-min)={spread}  (0 => identical elevation count across variants)\n")

    rows_out: List[Dict[str, Any]] = []
    for idx in range(len(chosen)):
        counts: List[int] = []
        for vn in vnames:
            s = all_summaries.get(vn, [])
            if idx < len(s):
                counts.append(s[idx][1])
        spread_val: Optional[int] = None
        if len(counts) == len(vnames) and counts:
            spread_val = max(counts) - min(counts)
        idx_obj = chosen[idx]
        attack_label = str(bde._normalize_label(df.loc[idx_obj, "label"]))
        row_slice = attack_label if slice_key_per_attack else slice_key
        for vn in vnames:
            s = all_summaries.get(vn, [])
            elev = s[idx][1] if idx < len(s) else None
            rows_out.append(
                {
                    "dataset": name,
                    "slice_key": row_slice,
                    "attack_label": attack_label,
                    "seed": seed,
                    "row_index": int(chosen[idx]),
                    "sample_order": idx + 1,
                    "variant": vn,
                    "n_elevated": elev,
                    "spread": spread_val,
                }
            )
    return rows_out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Grouping ablation: coarse / baseline / fine")
    p.add_argument("--dataset", choices=("nsl_kdd", "netml", "both"), default="both")
    p.add_argument(
        "--scope",
        choices=("sample", "all", "by_label"),
        default="sample",
        help="sample: random N anomalies; all: every non-normal row; by_label: one run per attack label",
    )
    p.add_argument(
        "--nsl-path",
        type=Path,
        default=None,
        help="Default: Dataset/KDDTest.csv (faster; use KDDTrain.csv if you need the full train set)",
    )
    p.add_argument("--netml-path", type=Path, default=None, help="Default: Dataset/netML_dataset.csv")
    p.add_argument("--netml-label", type=str, default=None)
    p.add_argument("--anomaly-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    def resolve(p: Optional[Path], default: Path) -> Path:
        if p is None:
            return default
        p = p.expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()

    nsl_path = resolve(args.nsl_path, _DEFAULT_NSL_KDD_CSV)
    netml_path = resolve(args.netml_path, _DEFAULT_NETML_CSV)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scope_tag = f"scope-{args.scope}" + (f"_anom{args.anomaly_samples}" if args.scope == "sample" else "")
    log_path = _RESULTS_DIR / f"grouping_ablation_{args.dataset}_{scope_tag}_seed{args.seed}.log"
    csv_path = _RESULTS_DIR / f"grouping_ablation_{args.dataset}_{scope_tag}_seed{args.seed}.csv"

    header = [
        "# grouping_ablation_experiment",
        f"# dataset(s)={args.dataset}",
        f"# scope={args.scope}",
        f"# seed={args.seed}",
        f"# anomaly_samples(used when scope=sample)={args.anomaly_samples}",
        "",
        "Variants:",
        "  coarse: fewer, merged behavior groups",
        "  baseline: current NSL config / NetML regex rules",
        "  fine: more groups (NSL splits rates/volume; NetML splits timing)",
        "",
    ]
    csv_rows: List[Dict[str, Any]] = []

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("\n".join(header) + "\n")

        if args.dataset in ("nsl_kdd", "both"):
            if not nsl_path.is_file():
                print(f"Missing NSL file: {nsl_path}", file=sys.stderr)
                return 1
            df = bde.load_nsl_kdd(nsl_path)
            logf.write(f"# nsl_path={nsl_path.resolve()} rows={len(df)}\n")

            def nsl_normal(lab: str) -> bool:
                return bde._normalize_label(lab) in NSL_NORMAL_LABELS

            variants = {
                "coarse": _nsl_coarse_groups(),
                "baseline": dict(NSL_BASELINE_GROUPS),
                "fine": _nsl_fine_groups(),
            }

            if args.scope == "sample":
                chosen = _pick_anomaly_indices(df, nsl_normal, args.anomaly_samples, args.seed)
                if chosen is None:
                    logf.write("NSL-KDD: no anomaly rows.\n")
                else:
                    csv_rows.extend(
                        _run_dataset(
                            "nsl_kdd", df, variants, nsl_normal, chosen, args.seed, logf, "sample", False
                        )
                    )
            elif args.scope == "all":
                chosen = _all_anomaly_indices(df, nsl_normal)
                if len(chosen) == 0:
                    logf.write("NSL-KDD: no anomaly rows.\n")
                else:
                    csv_rows.extend(
                        _run_dataset(
                            "nsl_kdd",
                            df,
                            variants,
                            nsl_normal,
                            chosen,
                            args.seed,
                            logf,
                            "__all__",
                            True,
                        )
                    )
            else:
                chosen = _all_anomaly_indices(df, nsl_normal)
                if len(chosen) == 0:
                    logf.write("NSL-KDD: no anomaly rows.\n")
                else:
                    logf.write(
                        "# NSL-KDD by_label: single pass over all anomalies; slice_key column = attack label\n"
                    )
                    part = _run_dataset(
                        "nsl_kdd",
                        df,
                        variants,
                        nsl_normal,
                        chosen,
                        args.seed,
                        logf,
                        "by_label",
                        True,
                        slice_key_per_attack=True,
                    )
                    csv_rows.extend(part)
                    _write_per_label_aggregates(logf, part)

        if args.dataset in ("netml", "both"):
            if not netml_path.is_file():
                print(f"Missing NetML file: {netml_path}", file=sys.stderr)
                return 1
            df = bde.load_netml_csv(netml_path, args.netml_label)
            logf.write(f"# netml_path={netml_path.resolve()} rows={len(df)}\n")

            net_pred = bde.default_netml_normal_predicate()
            variants = {
                "coarse": _netml_coarse_groups(df),
                "baseline": _netml_baseline_groups(df),
                "fine": _netml_fine_groups(df),
            }

            if args.scope == "sample":
                chosen = _pick_anomaly_indices(df, net_pred, args.anomaly_samples, args.seed)
                if chosen is None:
                    logf.write("NetML: no anomaly rows.\n")
                else:
                    csv_rows.extend(
                        _run_dataset(
                            "netml", df, variants, net_pred, chosen, args.seed, logf, "sample", False
                        )
                    )
            elif args.scope == "all":
                chosen = _all_anomaly_indices(df, net_pred)
                if len(chosen) == 0:
                    logf.write("NetML: no anomaly rows.\n")
                else:
                    csv_rows.extend(
                        _run_dataset(
                            "netml",
                            df,
                            variants,
                            net_pred,
                            chosen,
                            args.seed,
                            logf,
                            "__all__",
                            True,
                        )
                    )
            else:
                chosen = _all_anomaly_indices(df, net_pred)
                if len(chosen) == 0:
                    logf.write("NetML: no anomaly rows.\n")
                else:
                    logf.write(
                        "# NetML by_label: single pass over all anomalies; slice_key column = attack label\n"
                    )
                    part = _run_dataset(
                        "netml",
                        df,
                        variants,
                        net_pred,
                        chosen,
                        args.seed,
                        logf,
                        "by_label",
                        True,
                        slice_key_per_attack=True,
                    )
                    csv_rows.extend(part)
                    _write_per_label_aggregates(logf, part)

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV written: {csv_path}", file=sys.stderr)

    print(f"Log written: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
