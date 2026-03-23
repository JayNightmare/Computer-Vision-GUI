"""
Retinex Image Enhancement — Main Runner
========================================
CLI entry point that loads an image, applies Retinex enhancement,
and saves the results.

Usage
-----
    python main.py
    python main.py --image image/images.jpg --mode msr
    python main.py --mode ssr --sigmas 80
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from retinex import apply_retinex
from utils import save_results


def _parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply Retinex-based image enhancement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default="image/images.jpg",
        help="Path to the input image (default: image/images.jpg).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ssr", "msr"],
        default="msr",
        help="Retinex mode: 'ssr' (Single-Scale) or 'msr' (Multi-Scale). Default: msr.",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        default="15,80,250",
        help="Comma-separated Gaussian sigma values (default: 15,80,250).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output/).",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate the Retinex enhancement pipeline."""
    args = _parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image not found at '{image_path}'.", file=sys.stderr)
        sys.exit(1)

    sigmas = [float(s.strip()) for s in args.sigmas.split(",")]

    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to read image '{image_path}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Applying {args.mode.upper()} Retinex (sigmas={sigmas}) ...")
    enhanced = apply_retinex(image, mode=args.mode, sigmas=sigmas)

    print(f"Saving results to '{args.output}/' ...")
    saved = save_results(
        original=image,
        enhanced=enhanced,
        output_dir=args.output,
        filename_stem=image_path.stem,
        mode=args.mode,
    )

    print("\nDone! Output files:")
    for label, path in saved.items():
        print(f"  {label:>12s}: {path}")


if __name__ == "__main__":
    main()
