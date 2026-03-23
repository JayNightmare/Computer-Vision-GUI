"""Image processing algorithms for Workshops 3 & 4."""

from .point_ops import (
    adjust_brightness,
    adjust_contrast,
    gamma_correction,
    histogram_equalization,
    apply_clahe,
    negative_transform,
    compute_histogram,
)

from .spatial_ops import (
    image_add,
    image_subtract,
    image_multiply,
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_average_filter,
    apply_median_filter,
    compute_mse,
    edge_sobel,
    edge_scharr,
    edge_laplacian,
    edge_roberts,
    edge_prewitt,
)

from .segmentation_ops import (
    adaptive_threshold,
    otsu_threshold,
    connected_components_analysis,
    contour_object_detection,
)

from .cnn_ops import (
    CLASS_NAMES,
    SyntheticConfig,
    TrainConfig,
    build_cnn_model,
    build_preview_samples,
    create_dataset,
    evaluate_model,
    normalized_to_uint8,
    predict_with_confidence,
    set_tf_logging_level,
    train_model,
)

__all__ = [
    # Workshop 3 — Point Processing
    "adjust_brightness",
    "adjust_contrast",
    "gamma_correction",
    "histogram_equalization",
    "apply_clahe",
    "negative_transform",
    "compute_histogram",
    # Workshop 4 — Spatial & Statistical Filtering
    "image_add",
    "image_subtract",
    "image_multiply",
    "add_gaussian_noise",
    "add_salt_pepper_noise",
    "apply_average_filter",
    "apply_median_filter",
    "compute_mse",
    "edge_sobel",
    "edge_scharr",
    "edge_laplacian",
    "edge_roberts",
    "edge_prewitt",
    # Workshop 5 - Image Segmentation
    "adaptive_threshold",
    "otsu_threshold",
    "connected_components_analysis",
    "contour_object_detection",
    # Workshop 1 - CNN Classification
    "CLASS_NAMES",
    "SyntheticConfig",
    "TrainConfig",
    "build_cnn_model",
    "build_preview_samples",
    "create_dataset",
    "evaluate_model",
    "normalized_to_uint8",
    "predict_with_confidence",
    "set_tf_logging_level",
    "train_model",
]
