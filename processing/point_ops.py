"""
Point Processing Operations
============================
Pure image-processing functions — no UI, no I/O.  All functions accept
and return NumPy arrays so they can be used from any front-end (CLI,
GUI, notebook).

Formulas follow Workshop 3 — Point Processing (Kingston University).
"""

from __future__ import annotations

import cv2
import numpy as np


def adjust_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """Adjust brightness by adding a bias to every pixel.

    Formula: g(x,y) = f(x,y) + β

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    beta : float
        Brightness offset (−100 … +100 typical range).
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def adjust_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Adjust contrast by scaling pixel intensities.

    Formula: g(x,y) = α · f(x,y)

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    alpha : float
        Gain factor (0.5 … 3.0 typical range).
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction via a look-up table.

    Formula: O = (I / 255)^(1/γ) × 255

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    gamma : float
        Gamma value (< 1 brightens, > 1 darkens).
    """
    if gamma <= 0:
        gamma = 0.01

    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization (operates on luminance channel).

    Converts to YCrCb, equalises the Y channel, and converts back.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Operates on the luminance channel of a YCrCb conversion.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    clip_limit : float
        Threshold for contrast limiting.
    tile_grid_size : tuple[int, int]
        Size of the local neighbourhood grid.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def negative_transform(image: np.ndarray) -> np.ndarray:
    """Compute the photographic negative.

    Formula: g(x,y) = 255 − f(x,y)

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    """
    return cv2.bitwise_not(image)


def compute_histogram(
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-channel histograms for a BGR or grayscale image.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image (3-channel) or grayscale uint8 image (2D).

    Returns
    -------
    dict[str, np.ndarray]
        For BGR: keys ``"b"``, ``"g"``, ``"r"`` each mapping to (256,).
        For grayscale: single key ``"gray"`` mapping to (256,).
    """
    if image.ndim == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        return {"gray": hist.flatten()}

    channels = {"b": 0, "g": 1, "r": 2}
    result: dict[str, np.ndarray] = {}
    for name, idx in channels.items():
        hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
        result[name] = hist.flatten()
    return result

