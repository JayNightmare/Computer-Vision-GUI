"""
Visualization Utilities
=======================
Side-by-side comparison figures and result saving for Retinex-enhanced
images.  Separated from the algorithm to keep display concerns out of
the core module.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_comparison(
    original: np.ndarray,
    enhanced: np.ndarray,
    title: str = "Retinex Enhancement",
) -> plt.Figure:
    """Create a side-by-side comparison figure.

    Parameters
    ----------
    original : np.ndarray
        Original BGR image (uint8).
    enhanced : np.ndarray
        Enhanced BGR image (uint8).
    title : str
        Super-title for the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The comparison figure (not shown; caller decides when to display/save).
    """
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(enhanced_rgb)
    axes[1].set_title("Enhanced (Retinex)")
    axes[1].axis("off")

    fig.tight_layout()
    return fig


def save_results(
    original: np.ndarray,
    enhanced: np.ndarray,
    output_dir: str | Path,
    filename_stem: str,
    mode: str = "msr",
) -> dict[str, Path]:
    """Save the enhanced image and comparison figure to disk.

    Parameters
    ----------
    original : np.ndarray
        Original BGR image.
    enhanced : np.ndarray
        Enhanced BGR image.
    output_dir : str | Path
        Directory to write outputs into (created if missing).
    filename_stem : str
        Base name without extension (e.g. ``"images"``).
    mode : str
        Retinex mode label used in the output filenames.

    Returns
    -------
    dict[str, Path]
        Mapping of output type to saved file path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    enhanced_path = output_path / f"{filename_stem}_{mode}_enhanced.jpg"
    comparison_path = output_path / f"{filename_stem}_{mode}_comparison.png"

    cv2.imwrite(str(enhanced_path), enhanced)

    fig = create_comparison(
        original,
        enhanced,
        title=f"{mode.upper()} Retinex Enhancement — {filename_stem}",
    )
    fig.savefig(str(comparison_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"enhanced": enhanced_path, "comparison": comparison_path}
