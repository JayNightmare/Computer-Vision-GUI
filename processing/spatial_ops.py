"""
Spatial & Statistical Filtering Operations
===========================================
Pure image-processing functions for Workshop 4 — no UI, no I/O.
All functions accept and return NumPy arrays.

Covers:
- Image arithmetic (add, subtract, multiply)
- Noise generation (Gaussian, Salt & Pepper)
- Spatial filtering (Average / Median)
- Edge detection (Sobel, Scharr, Laplacian, Roberts, Prewitt)
- Error metric (MSE)
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image Arithmetic
# ---------------------------------------------------------------------------

def image_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Saturated pixel-wise addition of two images.

    Parameters
    ----------
    a, b : np.ndarray
        Images of identical shape and dtype (uint8).
    """
    return cv2.add(a, b)


def image_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Saturated pixel-wise subtraction (a − b).

    Parameters
    ----------
    a, b : np.ndarray
        Images of identical shape and dtype (uint8).
    """
    return cv2.subtract(a, b)


def image_multiply(a: np.ndarray, b: np.ndarray, scale: float = 1.0 / 255) -> np.ndarray:
    """Pixel-wise multiplication with normalization.

    Result = saturate_cast<uint8>(a * b * scale)

    Parameters
    ----------
    a, b : np.ndarray
        Images of identical shape and dtype (uint8).
    scale : float
        Normalization factor (default 1/255 keeps values in 0–255).
    """
    return cv2.multiply(a, b, scale=scale)


# ---------------------------------------------------------------------------
# Noise Generation
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 20.0,
) -> np.ndarray:
    """Add Gaussian noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input uint8 image.
    mean : float
        Mean of the Gaussian distribution.
    std : float
        Standard deviation of the Gaussian distribution.
    """
    noise = np.random.normal(mean, std, image.shape).astype(np.float64)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """Add salt-and-pepper noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Input uint8 image.
    amount : float
        Fraction of pixels affected (0–1).
    """
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(total_pixels * amount / 2)
    num_pepper = int(total_pixels * amount / 2)

    # Salt (white pixels)
    coords = [np.random.randint(0, d, num_salt) for d in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper (black pixels)
    coords = [np.random.randint(0, d, num_pepper) for d in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy


# ---------------------------------------------------------------------------
# Spatial Filters
# ---------------------------------------------------------------------------

def apply_average_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply a mean (box) filter.

    Parameters
    ----------
    image : np.ndarray
        Input uint8 image.
    ksize : int
        Kernel side length (must be odd and ≥ 1).
    """
    return cv2.blur(image, (ksize, ksize))


def apply_median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply a median filter.

    Parameters
    ----------
    image : np.ndarray
        Input uint8 image.
    ksize : int
        Kernel side length (must be odd and ≥ 1).
    """
    return cv2.medianBlur(image, ksize)


# ---------------------------------------------------------------------------
# Error Metric
# ---------------------------------------------------------------------------

def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error between two images.

    Parameters
    ----------
    a, b : np.ndarray
        Images of identical shape.

    Returns
    -------
    float
        The MSE value (lower is better; 0 = identical).
    """
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# Edge Detection
# ---------------------------------------------------------------------------

def edge_sobel(image: np.ndarray) -> np.ndarray:
    """Sobel edge detection via OpenCV.

    Computes gradient magnitude from X and Y Sobel derivatives.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    """
    gray = _ensure_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_scharr(image: np.ndarray) -> np.ndarray:
    """Scharr edge detection via OpenCV.

    Uses the Scharr 3×3 operator (more accurate than Sobel for small kernels).

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    """
    gray = _ensure_gray(image)
    gx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_laplacian(image: np.ndarray) -> np.ndarray:
    """Laplacian edge detection via OpenCV (2nd derivative).

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    """
    gray = _ensure_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_roberts(image: np.ndarray) -> np.ndarray:
    """Roberts Cross edge detection via custom 2×2 kernels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    """
    gray = _ensure_gray(image)
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    gx = cv2.filter2D(gray.astype(np.float64), -1, kernel_x)
    gy = cv2.filter2D(gray.astype(np.float64), -1, kernel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edge_prewitt(image: np.ndarray) -> np.ndarray:
    """Prewitt edge detection via custom 3×3 kernels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image.
    """
    gray = _ensure_gray(image)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = cv2.filter2D(gray.astype(np.float64), -1, kernel_x)
    gy = cv2.filter2D(gray.astype(np.float64), -1, kernel_y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if the image has 3+ channels."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
