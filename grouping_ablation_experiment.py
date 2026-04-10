#!/usr/bin/env python3
"""
Feature-grouping sensitivity (ablation): coarse / baseline / fine behavior maps.

Runs the same anomaly rows three times with different groupings, logs full outputs
plus a short comparison (elevated-behavior counts and side-by-side summaries).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
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


def _capture_run(
    df,
    groups: Mapping[str, List[str]],
    normal_pred: Callable,
    chosen: np.ndarray,
    seed: int,
) -> Tuple[str, List[Tuple[str, int]]]:
    """Returns stdout capture and list of (sample_key, n_elevated) per anomaly row."""
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
    summaries: List[Tuple[str, int]] = []
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


def _run_dataset(
    name: str,
    df,
    variants: Dict[VariantName, Mapping[str, List[str]]],
    normal_pred: Callable,
    chosen: np.ndarray,
    seed: int,
    logf,
) -> List[Dict[str, Any]]:
    logf.write(f"\n{'=' * 72}\n")
    logf.write(f"# Dataset: {name}\n")
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

        block, summ = _capture_run(df, groups, normal_pred, chosen, seed)
        logf.write(block)
        if not summ:
            logf.write("(no per-sample summary parsed)\n")
        all_summaries[vname] = summ

    logf.write(f"\n{'=' * 72}\n")
    logf.write("# COMPARISON (same rows across variants)\n")
    logf.write("# n_elevated = count of behavior groups with ↑ or ↑↑ for that sample.\n")
    logf.write(f"{'=' * 72}\n")
    vnames = list(variants.keys())
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
        for vn in vnames:
            s = all_summaries.get(vn, [])
            elev = s[idx][1] if idx < len(s) else None
            rows_out.append(
                {
                    "dataset": name,
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _RESULTS_DIR / f"grouping_ablation_{args.dataset}_seed{args.seed}_anom{args.anomaly_samples}_{stamp}.log"
    csv_path = _RESULTS_DIR / f"grouping_ablation_{args.dataset}_seed{args.seed}_anom{args.anomaly_samples}_{stamp}.csv"

    header = [
        "# grouping_ablation_experiment",
        f"# dataset(s)={args.dataset}",
        f"# seed={args.seed}",
        f"# anomaly_samples={args.anomaly_samples}",
        f"# time={datetime.now().isoformat(timespec='seconds')}",
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

            chosen = _pick_anomaly_indices(df, nsl_normal, args.anomaly_samples, args.seed)
            if chosen is None:
                logf.write("NSL-KDD: no anomaly rows.\n")
            else:
                variants = {
                    "coarse": _nsl_coarse_groups(),
                    "baseline": dict(NSL_BASELINE_GROUPS),
                    "fine": _nsl_fine_groups(),
                }
                csv_rows.extend(_run_dataset("nsl_kdd", df, variants, nsl_normal, chosen, args.seed, logf))

        if args.dataset in ("netml", "both"):
            if not netml_path.is_file():
                print(f"Missing NetML file: {netml_path}", file=sys.stderr)
                return 1
            df = bde.load_netml_csv(netml_path, args.netml_label)
            logf.write(f"# netml_path={netml_path.resolve()} rows={len(df)}\n")

            net_pred = bde.default_netml_normal_predicate()
            chosen = _pick_anomaly_indices(df, net_pred, args.anomaly_samples, args.seed)
            if chosen is None:
                logf.write("NetML: no anomaly rows.\n")
            else:
                variants = {
                    "coarse": _netml_coarse_groups(df),
                    "baseline": _netml_baseline_groups(df),
                    "fine": _netml_fine_groups(df),
                }
                csv_rows.extend(_run_dataset("netml", df, variants, net_pred, chosen, args.seed, logf))

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV written: {csv_path}", file=sys.stderr)

    print(f"Log written: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
