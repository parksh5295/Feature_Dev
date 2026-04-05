#!/usr/bin/env python3
"""Convert NSL-KDD KDDTrain+.txt and KDDTest+.txt to two CSV files (no split)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.nsl_kdd_config import NSL_KDD_COLUMNS, NSL_OPTIONAL_DIFFICULTY


def column_names_for_ncols(ncols: int) -> List[str]:
    base = len(NSL_KDD_COLUMNS)
    if ncols == base:
        return list(NSL_KDD_COLUMNS)
    if ncols == base + 1:
        return list(NSL_KDD_COLUMNS) + [NSL_OPTIONAL_DIFFICULTY]
    raise ValueError(
        f"Expected {base} or {base + 1} columns (NSL-KDD+ / +difficulty), got {ncols}"
    )


def txt_to_csv(txt_path: str, csv_path: str, write_header: bool) -> int:
    df = pd.read_csv(txt_path, header=None, low_memory=False)
    ncols = df.shape[1]
    df.columns = column_names_for_ncols(ncols)
    df.to_csv(csv_path, index=False, header=write_header)
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert KDDTrain+.txt and KDDTest+.txt to KDDTrain.csv and KDDTest.csv",
    )
    parser.add_argument(
        "--train-txt",
        default=os.path.join(_ROOT, "Dataset", "KDDTrain+.txt"),
        help="Path to KDDTrain+.txt",
    )
    parser.add_argument(
        "--test-txt",
        default=os.path.join(_ROOT, "Dataset", "KDDTest+.txt"),
        help="Path to KDDTest+.txt",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_ROOT, "Dataset"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Write CSV without a header row (values only, like .txt but comma-separated)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.train_txt):
        print(f"Error: train file not found: {args.train_txt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.test_txt):
        print(f"Error: test file not found: {args.test_txt}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    write_header = not args.no_header

    out_train = os.path.join(args.out_dir, "KDDTrain.csv")
    out_test = os.path.join(args.out_dir, "KDDTest.csv")

    n_train = txt_to_csv(args.train_txt, out_train, write_header)
    n_test = txt_to_csv(args.test_txt, out_test, write_header)

    print(f"Wrote {out_train}  ({n_train} rows, header={write_header})")
    print(f"Wrote {out_test}   ({n_test} rows, header={write_header})")


if __name__ == "__main__":
    main()
