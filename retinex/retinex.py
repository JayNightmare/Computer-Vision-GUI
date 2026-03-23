"""
Retinex Algorithm Module
========================
Pure algorithm implementation of Single-Scale Retinex (SSR) and
Multi-Scale Retinex (MSR) for image enhancement.

This module contains NO I/O or display logic — only the mathematical
transformations.  All functions operate on NumPy arrays.

References
----------
- Land, E. H. (1977). The Retinex Theory of Color Vision.
- Jobson, D. J. et al. (1997). A Multiscale Retinex for Bridging the Gap
  Between Color Images and the Human Observation of Scenes.
"""

from __future__ import annotations

import numpy as np
import cv2

DEFAULT_SIGMAS: list[float] = [15.0, 80.0, 250.0]
_EPSILON: float = 1.0  # Prevents log(0); added before taking logarithm.


def single_scale_retinex(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Single-Scale Retinex (SSR) to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image as a float64 array (any number of channels).
    sigma : float
        Standard deviation of the Gaussian kernel used to estimate
        the illumination component.

    Returns
    -------
    np.ndarray
        SSR-enhanced image in log-space (not yet normalised to 0-255).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    retinex = np.log10(image + _EPSILON) - np.log10(blurred + _EPSILON)
    return retinex


def multi_scale_retinex(
    image: np.ndarray,
    sigmas: list[float] | None = None,
) -> np.ndarray:
    """Apply Multi-Scale Retinex (MSR) to an image.

    Averages SSR results across multiple Gaussian scales to balance
    local contrast enhancement with dynamic-range compression.

    Parameters
    ----------
    image : np.ndarray
        Input image as a float64 array.
    sigmas : list[float] | None
        Gaussian sigma values for each scale.
        Defaults to ``[15, 80, 250]`` (Jobson et al., 1997).

    Returns
    -------
    np.ndarray
        MSR-enhanced image in log-space.
    """
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS

    retinex = np.zeros_like(image, dtype=np.float64)
    for sigma in sigmas:
        retinex += single_scale_retinex(image, sigma)

    retinex /= len(sigmas)
    return retinex


def _normalise_to_uint8(image: np.ndarray) -> np.ndarray:
    """Linearly stretch a float image into the 0-255 uint8 range.

    Parameters
    ----------
    image : np.ndarray
        Arbitrary-range float image.

    Returns
    -------
    np.ndarray
        Image normalised to ``[0, 255]`` as ``uint8``.
    """
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    normalised = (image - min_val) / (max_val - min_val) * 255.0
    return np.clip(normalised, 0, 255).astype(np.uint8)


def apply_retinex(
    image: np.ndarray,
    mode: str = "msr",
    sigmas: list[float] | None = None,
) -> np.ndarray:
    """Public API — apply Retinex enhancement to a BGR/RGB uint8 image.

    Parameters
    ----------
    image : np.ndarray
        Input image in ``uint8`` format (BGR or RGB, 1- or 3-channel).
    mode : str
        ``"ssr"`` for Single-Scale Retinex or ``"msr"`` for Multi-Scale
        Retinex.
    sigmas : list[float] | None
        Gaussian sigma value(s).  For SSR only the first value is used.
        Defaults to ``[15, 80, 250]``.

    Returns
    -------
    np.ndarray
        Enhanced image as ``uint8`` in the same channel layout as the input.

    Raises
    ------
    ValueError
        If *mode* is not ``"ssr"`` or ``"msr"``.
    """
    if mode not in ("ssr", "msr"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'ssr' or 'msr'.")

    if sigmas is None:
        sigmas = DEFAULT_SIGMAS

    img_float = image.astype(np.float64)

    if mode == "ssr":
        retinex = single_scale_retinex(img_float, sigmas[0])
    else:
        retinex = multi_scale_retinex(img_float, sigmas)

    return _normalise_to_uint8(retinex)
