"""
Segmentation and Object Detection Operations
===========================================
Pure image-processing functions for Workshop 5.
"""

from __future__ import annotations

import cv2
import numpy as np


def adaptive_threshold(
    image: np.ndarray,
    method: str = "gaussian",
    block_size: int = 11,
    c_value: float = 2.0,
) -> np.ndarray:
    """Compute adaptive thresholding on a grayscale-converted image."""
    gray = _ensure_gray(image)
    block_size = _ensure_odd(max(int(block_size), 3))

    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method.lower() == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    return cv2.adaptiveThreshold(
        gray,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        float(c_value),
    )


def otsu_threshold(image: np.ndarray, blur_ksize: int = 0) -> tuple[float, np.ndarray]:
    """Compute Otsu binary threshold, optionally after Gaussian blur."""
    gray = _ensure_gray(image)
    if int(blur_ksize) > 0:
        ksize = _ensure_odd(max(int(blur_ksize), 3))
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return float(threshold), binary


def connected_components_analysis(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Label connected components and return color labels, annotated image, and measurements."""
    binary_u8 = _to_binary_uint8(binary)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        binary_u8,
        connectivity=8,
    )

    colored_labels = _colorize_labels(labels_im, num_labels)
    annotated = colored_labels.copy()

    measurements: list[dict] = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        cx, cy = centroids[label_id]

        region_mask = (labels_im == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        perimeter = 0.0
        if contours:
            perimeter = float(cv2.arcLength(contours[0], True))

        measurements.append(
            {
                "Label": int(label_id),
                "Area": int(area),
                "Perimeter": float(perimeter),
                "BBox": (int(x), int(y), int(w), int(h)),
                "Centroid": (float(cx), float(cy)),
            }
        )

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(
            annotated,
            f"{label_id}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return colored_labels, annotated, measurements


def contour_object_detection(
    image: np.ndarray,
    blur_ksize: int = 5,
    invert: bool = False,
    min_area: float = 60.0,
    min_circularity: float = 0.0,
    morph_open_ksize: int = 3,
    morph_close_ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Detect objects using contour analysis with area and circularity filtering."""
    gray = _ensure_gray(image)

    blur_ksize = _ensure_odd(max(int(blur_ksize), 3))
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    thresh_mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(blurred, 0, 255, thresh_mode | cv2.THRESH_OTSU)

    morph_open_ksize = _ensure_odd(max(int(morph_open_ksize), 1))
    morph_close_ksize = _ensure_odd(max(int(morph_close_ksize), 1))

    if morph_open_ksize > 1:
        open_kernel = np.ones((morph_open_ksize, morph_open_ksize), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    if morph_close_ksize > 1:
        close_kernel = np.ones((morph_close_ksize, morph_close_ksize), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    measurements: list[dict] = []

    next_label = 1
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area):
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue

        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter))
        if circularity < float(min_circularity):
            continue

        x, y, w, h = cv2.boundingRect(contour)

        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
        else:
            cx = float(x + w / 2.0)
            cy = float(y + h / 2.0)

        measurements.append(
            {
                "Label": int(next_label),
                "Area": area,
                "Perimeter": perimeter,
                "Circularity": circularity,
                "BBox": (int(x), int(y), int(w), int(h)),
                "Centroid": (cx, cy),
            }
        )

        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 200, 0), 1)
        cv2.putText(
            annotated,
            f"ID {next_label}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        next_label += 1

    return binary, annotated, measurements


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _to_binary_uint8(binary: np.ndarray) -> np.ndarray:
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)
    return np.where(binary > 0, 255, 0).astype(np.uint8)


def _colorize_labels(labels_im: np.ndarray, num_labels: int) -> np.ndarray:
    label_hue = np.uint8(179 * labels_im / max(num_labels, 1))
    blank_channel = np.full_like(label_hue, 255)

    labeled_hsv = np.dstack((label_hue, blank_channel, blank_channel))
    colored = cv2.cvtColor(labeled_hsv, cv2.COLOR_HSV2BGR)
    colored[label_hue == 0] = 0
    return colored
