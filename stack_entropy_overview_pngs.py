#!/usr/bin/env python3
"""
Vertically stack NSL-KDD and NetML entropy overview PNGs into one figure.

PNG stacking is lossless (no JPEG). This script avoids dithering on palette
images, composites RGBA on white explicitly, preserves DPI metadata, and uses
nearest-neighbor scaling when --scale is an integer > 1 (sharp edges for print).

Example:
  python stack_entropy_overview_pngs.py
  python stack_entropy_overview_pngs.py --gap 24 --scale 2
  python stack_entropy_overview_pngs.py --dpi 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

_ROOT = Path(__file__).resolve().parent
_DEFAULT_NSL = _ROOT / "results" / "fig_entropy_overview_nsl_kdd.png"
_DEFAULT_NETML = _ROOT / "results" / "fig_entropy_overview_netml.png"
_DEFAULT_OUT = _ROOT / "results" / "fig_entropy_overview_nsl_netml_stacked.png"


def _to_rgb_flat(im) -> "Image.Image":
    """RGB 8-bit, no dither; RGBA flattened onto white."""
    from PIL import Image

    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    if im.mode == "RGB":
        return im.copy()
    return im.convert("RGB", dither=Image.Dither.NONE)


def _read_png(path: Path):
    from PIL import Image

    im = Image.open(path)
    im.load()
    return im


def _output_dpi(first_im, override: float | None) -> Tuple[float, float] | None:
    if override is not None and override > 0:
        return (float(override), float(override))
    dpi = first_im.info.get("dpi")
    if dpi is None:
        return None
    if isinstance(dpi, tuple) and len(dpi) >= 2:
        x, y = float(dpi[0]), float(dpi[1])
        if x > 0 and y > 0:
            return (x, y)
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stack entropy overview PNGs vertically (lossless)")
    p.add_argument("--nsl", type=Path, default=_DEFAULT_NSL, help="Top image (NSL-KDD)")
    p.add_argument("--netml", type=Path, default=_DEFAULT_NETML, help="Bottom image (NetML)")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="Output PNG path")
    p.add_argument("--gap", type=int, default=16, help="Pixels of white space between images")
    p.add_argument(
        "--order",
        choices=("nsl_netml", "netml_nsl"),
        default="nsl_netml",
        help="Vertical order: nsl_netml = NSL on top",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Integer upscale factor (>=1). Uses NEAREST so lines stay sharp (e.g. 2 for 2x pixels).",
    )
    p.add_argument(
        "--dpi",
        type=float,
        default=None,
        help="Write this DPI into output PNG (default: copy from first input if present)",
    )
    args = p.parse_args(argv)

    try:
        from PIL import Image
    except ImportError:
        print("Install Pillow: pip install Pillow", file=sys.stderr)
        return 1

    if args.scale < 1:
        print("--scale must be >= 1", file=sys.stderr)
        return 1

    paths = (args.nsl, args.netml) if args.order == "nsl_netml" else (args.netml, args.nsl)
    for path in paths:
        if not path.is_file():
            print(f"Not found: {path}", file=sys.stderr)
            return 1

    raw_list = [_read_png(p) for p in paths]
    imgs = [_to_rgb_flat(im) for im in raw_list]

    if args.scale > 1:
        resample = Image.Resampling.NEAREST
        imgs = [
            im.resize((im.width * args.scale, im.height * args.scale), resample, reducing_gap=None)
            for im in imgs
        ]

    w = max(im.width for im in imgs)
    gap = max(0, args.gap) * args.scale
    h = sum(im.height for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for im in imgs:
        x = (w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + gap

    dpi_tuple = _output_dpi(raw_list[0], args.dpi)
    save_kw = {
        "format": "PNG",
        "optimize": False,
        "compress_level": 3,
    }
    if dpi_tuple is not None:
        save_kw["dpi"] = dpi_tuple

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out, **save_kw)
    print(f"Wrote {args.out}" + (f" dpi={dpi_tuple}" if dpi_tuple else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
