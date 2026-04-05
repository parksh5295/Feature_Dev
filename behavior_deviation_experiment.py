#!/usr/bin/env python3
"""Behavior-level deviation from normal feature means (NSL-KDD, NetML CSV)."""

from __future__ import annotations

import argparse
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple

import numpy as np
import pandas as pd

from utils.nsl_kdd_config import (
    NSL_KDD_BEHAVIOR_GROUPS,
    NSL_KDD_COLUMNS,
    NSL_NORMAL_LABELS,
    NSL_OPTIONAL_DIFFICULTY,
)

_ROOT = Path(__file__).resolve().parent
_DEFAULT_DATASET_DIR = _ROOT / "Dataset"
_DEFAULT_NSL_KDD_CSV = _DEFAULT_DATASET_DIR / "KDDTrain.csv"
_DEFAULT_NETML_CSV = _DEFAULT_DATASET_DIR / "netML_dataset.csv"
_RESULTS_DIR = _ROOT / "results"


class _TeeStdout:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


@contextmanager
def _tee_stdout_to_file(log_file: TextIO) -> Iterator[None]:
    prev = sys.stdout
    sys.stdout = _TeeStdout(prev, log_file)
    try:
        yield
    finally:
        sys.stdout = prev


# First matching rule wins; used when column names vary across NetML exports.
NETML_GROUP_RULES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"iat|inter[_-]?arrival|idle", re.I), "Timing pattern"),
    (re.compile(r"duration|flow_?dur|time", re.I), "Timing pattern"),
    (re.compile(r"byte|octet|length|size|pkt|packet|len", re.I), "Data volume"),
    (re.compile(r"tcp|udp|icmp|protocol|flag|fin|rst|syn|ack|psh|urg", re.I), "Protocol / flags"),
    (re.compile(r"count|tot_|total_|num_|srv_|host_|flow", re.I), "Connection intensity"),
]


def _infer_netml_behavior(column: str) -> str:
    for pat, name in NETML_GROUP_RULES:
        if pat.search(column):
            return name
    return "Other features"


def _normalize_label(s: str) -> str:
    return str(s).strip().lower()


def _drop_unnamed_index_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip pandas export index columns (e.g. Unnamed: 0) from flow CSVs."""
    keep = [c for c in df.columns if not str(c).strip().lower().startswith("unnamed")]
    return df[keep]


def load_nsl_kdd(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if not first:
        raise ValueError(f"빈 파일: {path}")
    tokens = first.split(",")
    first_field = tokens[0].strip().lower()
    # Official .txt has no header; CSV exports often start with "duration".
    has_header = first_field == "duration"
    df = pd.read_csv(path, header=0 if has_header else None, low_memory=False)
    if not has_header:
        n = df.shape[1]
        if n == len(NSL_KDD_COLUMNS) + 1:
            df.columns = list(NSL_KDD_COLUMNS) + [NSL_OPTIONAL_DIFFICULTY]
        elif n == len(NSL_KDD_COLUMNS):
            df.columns = list(NSL_KDD_COLUMNS)
        else:
            raise ValueError(
                f"NSL-KDD 열 개수 불일치: 기대 {len(NSL_KDD_COLUMNS)} 또는 +1, 실제 {n}"
            )
    df.columns = [str(c).strip() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError("NSL-KDD에 label 컬럼이 없습니다.")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def load_netml_csv(path: Path, label_column: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    lc = label_column
    if lc is None:
        for cand in ("label", "Label", "class", "Class", "y", "target", "attack_cat", "Attack"):
            if cand in df.columns:
                lc = cand
                break
    if lc is None or lc not in df.columns:
        raise ValueError(
            "NetML CSV에서 라벨 컬럼을 찾을 수 없습니다. --netml-label 로 지정하세요."
        )
    df = df.rename(columns={lc: "label"})
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = _drop_unnamed_index_columns(df)
    return df


def numeric_feature_columns(
    df: pd.DataFrame,
    label_col: str = "label",
    extra_exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    exclude = {label_col, NSL_OPTIONAL_DIFFICULTY, "difficulty"}
    if extra_exclude:
        exclude.update(extra_exclude)
    num_cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
    return num_cols


def prepare_numeric_frame(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce to numeric; map inf to NaN; impute with column medians then 0 if still NaN."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X = X.fillna(0.0)
    usable = [c for c in feature_cols if c in X.columns and X[c].notna().all()]
    return X[usable], usable


def compute_normal_mean(X_normal: pd.DataFrame) -> pd.Series:
    return X_normal.mean(axis=0)


def feature_deviations_row(row: pd.Series, mu: pd.Series) -> pd.Series:
    aligned = mu.reindex(row.index)
    return (row - aligned).abs()


def behavior_scores_from_deviations(
    deviations: pd.Series, groups: Mapping[str, List[str]]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for bname, feats in groups.items():
        present = [f for f in feats if f in deviations.index]
        if not present:
            continue
        out[bname] = float(deviations[present].mean())
    return out


def fit_behavior_thresholds(
    normal_scores: pd.DataFrame,
    quantile_high: float = 0.90,
    quantile_very_high: float = 0.99,
) -> Tuple[pd.Series, pd.Series]:
    """Per-behavior quantiles on normal traffic scores."""
    q_hi = normal_scores.quantile(quantile_high)
    q_vhi = normal_scores.quantile(quantile_very_high)
    return q_hi, q_vhi


def label_behavior_level(
    score: float,
    q_hi: float,
    q_vhi: float,
) -> str:
    if score >= q_vhi:
        return "↑↑"
    if score >= q_hi:
        return "↑"
    return "normal"


def format_feature_explanation(deviations: pd.Series, top_k: int = 6) -> str:
    s = deviations.sort_values(ascending=False).head(top_k)
    parts = [f"{k} dev={v:.4g}" for k, v in s.items()]
    return ", ".join(parts)


def format_behavior_explanation(
    scores: Mapping[str, float],
    q_hi: pd.Series,
    q_vhi: pd.Series,
) -> str:
    lines = []
    for b in sorted(scores.keys()):
        sym = label_behavior_level(scores[b], float(q_hi.get(b, np.inf)), float(q_vhi.get(b, np.inf)))
        lines.append(f"  {b}: {scores[b]:.4g} → {sym}")
    return "\n".join(lines)


def run_experiment(
    df: pd.DataFrame,
    groups: Mapping[str, List[str]],
    normal_label_predicate,
    anomaly_sample_limit: int = 5,
    random_state: int = 42,
) -> None:
    feature_cols = numeric_feature_columns(df)
    grouped_feats = set()
    for fs in groups.values():
        grouped_feats.update(fs)
    feature_cols = [c for c in feature_cols if c in grouped_feats]

    X_all, cols = prepare_numeric_frame(df, feature_cols)
    df = df.loc[X_all.index].copy()
    X_all.columns = cols

    labels = df["label"].map(_normalize_label)
    normal_mask = labels.map(normal_label_predicate)
    if not normal_mask.any():
        raise ValueError("정상(normal) 표본이 0개입니다. 라벨 이름을 확인하세요.")

    X_normal = X_all.loc[normal_mask]
    mu = compute_normal_mean(X_normal)

    normal_behavior_rows: List[Dict[str, float]] = []
    for _, row in X_normal.iterrows():
        dev = feature_deviations_row(row, mu)
        normal_behavior_rows.append(behavior_scores_from_deviations(dev, groups))
    normal_scores = pd.DataFrame(normal_behavior_rows).fillna(0.0)
    q_hi, q_vhi = fit_behavior_thresholds(normal_scores)

    anomaly_mask = ~normal_mask
    anomaly_idx = df.index[anomaly_mask]
    if len(anomaly_idx) == 0:
        print("이상 표본이 없습니다.")
        return

    rng = np.random.default_rng(random_state)
    pick_n = min(anomaly_sample_limit, len(anomaly_idx))
    chosen = rng.choice(anomaly_idx.to_numpy(), size=pick_n, replace=False)

    print("=== 정상 기준: feature 평균(일부) ===")
    print(mu.head(12).to_string())
    print("\n=== Behavior score 분위수 (정상 분포 기준) ===")
    for b in sorted(q_hi.index):
        print(f"  {b}: p90={q_hi[b]:.4g}, p99={q_vhi[b]:.4g}")

    print("\n=== 이상 샘플 예시: feature-level vs behavior-level ===")
    for i, idx in enumerate(chosen, 1):
        row = X_all.loc[idx]
        lbl = df.loc[idx, "label"]
        dev = feature_deviations_row(row, mu)
        bscores = behavior_scores_from_deviations(dev, groups)
        print(f"\n--- 샘플 {i} | label={lbl} ---")
        print("Feature explanation (상위 deviation):")
        print(" ", format_feature_explanation(dev))
        print("Proposed (behavior deviation):")
        print(format_behavior_explanation(bscores, q_hi, q_vhi))


def default_netml_normal_predicate() -> callable:
    normal_tokens = frozenset(
        {
            "benign",
            "normal",
            "background",
            "none",
            "non",
            "non-attack",
            "nonattack",
        }
    )

    def pred(lab: str) -> bool:
        s = _normalize_label(lab)
        if s in normal_tokens:
            return True
        if "benign" in s or "normal" in s:
            return True
        return False

    return pred


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Behavior-level deviation 실험 (NSL-KDD / NetML)")
    p.add_argument(
        "--dataset",
        choices=("nsl_kdd", "netml"),
        required=True,
        help="데이터셋 종류",
    )
    p.add_argument(
        "--path",
        type=Path,
        default=None,
        help=(
            "CSV or NSL-KDD .txt path. Default: Dataset/KDDTrain.csv (nsl_kdd) or "
            "Dataset/netML_dataset.csv (netml) next to this script."
        ),
    )
    p.add_argument("--netml-label", type=str, default=None, help="NetML 라벨 컬럼명")
    p.add_argument("--anomaly-samples", type=int, default=5, help="출력할 이상 샘플 개수")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.path is None:
        path = _DEFAULT_NSL_KDD_CSV if args.dataset == "nsl_kdd" else _DEFAULT_NETML_CSV
    else:
        path = args.path.expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

    if not path.is_file():
        print(f"파일 없음: {path}", file=sys.stderr)
        return 1

    if args.dataset == "nsl_kdd":
        df = load_nsl_kdd(path)
        groups = NSL_KDD_BEHAVIOR_GROUPS

        def normal_pred(lab: str) -> bool:
            return _normalize_label(lab) in NSL_NORMAL_LABELS

    else:
        df = load_netml_csv(path, args.netml_label)
        num_cols = numeric_feature_columns(df)
        groups = {}
        for c in num_cols:
            b = _infer_netml_behavior(c)
            groups.setdefault(b, []).append(c)
        groups = {k: v for k, v in groups.items() if v}
        normal_pred = default_netml_normal_predicate()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        _RESULTS_DIR
        / f"{args.dataset}_seed{args.seed}_anom{args.anomaly_samples}_{stamp}.log"
    )
    header_lines = [
        "# behavior_deviation_experiment",
        f"# dataset={args.dataset}",
        f"# seed={args.seed}",
        f"# path={path.resolve()}",
        f"# anomaly_samples={args.anomaly_samples}",
        f"# rows={len(df)}",
        f"# time={datetime.now().isoformat(timespec='seconds')}",
    ]
    if args.dataset == "netml":
        header_lines.append(f"# netml_label={args.netml_label or '(auto)'}")

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("\n".join(header_lines) + "\n\n")
        with _tee_stdout_to_file(logf):
            run_experiment(
                df,
                groups,
                normal_pred,
                anomaly_sample_limit=args.anomaly_samples,
                random_state=args.seed,
            )

    print(f"Log written: {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
