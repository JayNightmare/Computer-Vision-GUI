"""CNN utilities for Workshop 1 synthetic shape classification."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

CLASS_NAMES = ("Circle", "Square", "Triangle", "Star")
NUM_CLASSES = 4

# Keep workshop console output focused on metrics/progress instead of TF internals.
tf.get_logger().setLevel(logging.ERROR)


def set_tf_logging_level(level: str = "ERROR") -> None:
    """Set TensorFlow logger verbosity level (ERROR, WARNING, INFO, DEBUG)."""

    normalized = str(level).strip().upper()
    level_map = {
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    tf.get_logger().setLevel(level_map.get(normalized, logging.ERROR))


@dataclass(slots=True)
class SyntheticConfig:
    """Configuration for synthetic shape dataset generation."""

    image_size: int = 28
    samples_per_class: int = 500
    test_size: float = 0.2
    apply_rotation: bool = True
    rotation_range: float = 18.0
    apply_noise: bool = True
    noise_std: float = 10.0
    seed: int = 42


@dataclass(slots=True)
class TrainConfig:
    """Training hyper-parameters for the CNN model."""

    epochs: int = 10
    batch_size: int = 32


def _get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def set_reproducible_seed(seed: int) -> None:
    """Set deterministic seeds for NumPy and TensorFlow."""

    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _apply_random_rotation(
    image: np.ndarray,
    image_size: int,
    rng: np.random.Generator,
    enabled: bool,
    rotation_range: float,
) -> np.ndarray:
    if not enabled:
        return image

    angle = float(rng.uniform(-rotation_range, rotation_range))
    matrix = cv2.getRotationMatrix2D((image_size / 2.0, image_size / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )


def _apply_random_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    enabled: bool,
    noise_std: float,
) -> np.ndarray:
    if not enabled:
        return image

    noise = rng.normal(0.0, noise_std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _generate_circle(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)
    max_radius = max(5, cfg.image_size // 3)
    radius = int(rng.integers(4, max_radius + 1))

    center_x = int(rng.integers(radius, cfg.image_size - radius))
    center_y = int(rng.integers(radius, cfg.image_size - radius))
    cv2.circle(image, (center_x, center_y), radius, 255, -1)

    image = _apply_random_rotation(
        image=image,
        image_size=cfg.image_size,
        rng=rng,
        enabled=cfg.apply_rotation,
        rotation_range=cfg.rotation_range,
    )
    image = _apply_random_noise(image, rng, cfg.apply_noise, cfg.noise_std)
    return image


def _generate_square(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)
    side = int(rng.integers(6, cfg.image_size - 5))
    x = int(rng.integers(0, cfg.image_size - side))
    y = int(rng.integers(0, cfg.image_size - side))
    cv2.rectangle(image, (x, y), (x + side, y + side), 255, -1)

    image = _apply_random_rotation(
        image=image,
        image_size=cfg.image_size,
        rng=rng,
        enabled=cfg.apply_rotation,
        rotation_range=cfg.rotation_range,
    )
    image = _apply_random_noise(image, rng, cfg.apply_noise, cfg.noise_std)
    return image

def _generate_triangle(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)
    side = int(rng.integers(6, cfg.image_size - 5))
    x = int(rng.integers(0, cfg.image_size - side))
    y = int(rng.integers(0, cfg.image_size - side))
    points = np.array([[x, y + side], [x + side // 2, y], [x + side, y + side]])
    cv2.fillPoly(image, [points], 255)

    image = _apply_random_rotation(
        image=image,
        image_size=cfg.image_size,
        rng=rng,
        enabled=cfg.apply_rotation,
        rotation_range=cfg.rotation_range,
    )
    image = _apply_random_noise(image, rng, cfg.apply_noise, cfg.noise_std)
    return image

def _generate_star(cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    image = np.zeros((cfg.image_size, cfg.image_size), dtype=np.uint8)
    side = int(rng.integers(6, cfg.image_size - 5))
    x = int(rng.integers(0, cfg.image_size - side))
    y = int(rng.integers(0, cfg.image_size - side))
    points = np.array(
        [
            [x + side // 2, y],
            [x + side * 3 // 4, y + side // 3],
            [x + side, y + side // 3],
            [x + side * 5 // 6, y + side * 2 // 3],
            [x + side * 7 // 8, y + side],
            [x + side // 2, y + side * 5 // 6],
            [x + side // 8, y + side],
            [x + side * 1 // 6, y + side * 2 // 3],
            [x, y + side // 3],
            [x + side // 4, y + side // 3],
        ]
    )
    cv2.fillPoly(image, [points], 255)

    image = _apply_random_rotation(
        image=image,
        image_size=cfg.image_size,
        rng=rng,
        enabled=cfg.apply_rotation,
        rotation_range=cfg.rotation_range,
    )
    image = _apply_random_noise(image, rng, cfg.apply_noise, cfg.noise_std)
    return image


def generate_shape_image(class_id: int, cfg: SyntheticConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate one synthetic image (uint8 grayscale) for a class id."""

    if class_id == 0:
        return _generate_circle(cfg, rng)
    elif class_id == 1:
        return _generate_square(cfg, rng)
    elif class_id == 2:
        return _generate_triangle(cfg, rng)
    else:
        return _generate_star(cfg, rng)


def create_dataset(
    cfg: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test dataset with one-hot labels and integer labels."""

    set_reproducible_seed(cfg.seed)
    rng = _get_rng(cfg.seed)

    images: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(cfg.samples_per_class):
        images.append(generate_shape_image(0, cfg, rng))
        labels.append(0)
        images.append(generate_shape_image(1, cfg, rng))
        labels.append(1)
        images.append(generate_shape_image(2, cfg, rng))
        labels.append(2)
        images.append(generate_shape_image(3, cfg, rng))
        labels.append(3)

    x = np.asarray(images, dtype=np.float32) / 255.0
    x = x.reshape(-1, cfg.image_size, cfg.image_size, 1)
    y_int = np.asarray(labels, dtype=np.int32)
    y = to_categorical(y_int, NUM_CLASSES)

    x_train, x_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
        x,
        y,
        y_int,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y_int,
    )

    return x_train, x_test, y_train, y_test, y_train_int, y_test_int


def build_cnn_model(image_size: int = 28) -> tf.keras.Model:
    """Build and compile a compact CNN model for binary shape classification."""

    tf.keras.backend.clear_session()
    model = Sequential(
        [
            tf.keras.Input(shape=(image_size, image_size, 1)),
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
) -> tf.keras.callbacks.History:
    """Train model and optionally stream epoch progress."""

    callbacks: list[tf.keras.callbacks.Callback] = []

    if progress_callback is not None:

        class EpochProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                normalized_logs = {
                    str(k): float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))
                }
                progress_callback(epoch + 1, cfg.epochs, normalized_logs)

        callbacks.append(EpochProgressCallback())

    return model.fit(
        x_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(x_val, y_val),
        verbose=0,
        callbacks=callbacks,
    )


def evaluate_model(model: tf.keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Evaluate model and return a metrics dictionary."""

    eval_result = model.evaluate(x_test, y_test, verbose=0)
    if isinstance(eval_result, (list, tuple)):
        if len(eval_result) >= 2:
            loss = float(eval_result[0])
            accuracy = float(eval_result[1])
        elif len(eval_result) == 1:
            loss = float(eval_result[0])
            accuracy = 0.0
        else:
            loss = 0.0
            accuracy = 0.0
    else:
        loss = float(eval_result)
        accuracy = 0.0
    return {"loss": loss, "accuracy": accuracy}


def predict_with_confidence(model: tf.keras.Model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict class ids and confidence scores for provided samples."""

    # Direct eager forward pass avoids repeatedly rebuilding Keras predict functions.
    probs = model(tf.convert_to_tensor(x), training=False).numpy()
    pred_ids = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    return pred_ids, confidence


def build_preview_samples(
    cfg: SyntheticConfig,
    count_per_class: int = 6,
    seed_offset: int = 777,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small sample set for visualization and confidence checks."""

    rng = _get_rng(cfg.seed + seed_offset)
    images: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(count_per_class):
        images.append(generate_shape_image(0, cfg, rng))
        labels.append(0)
        images.append(generate_shape_image(1, cfg, rng))
        labels.append(1)
        images.append(generate_shape_image(2, cfg, rng))
        labels.append(2)
        images.append(generate_shape_image(3, cfg, rng))
        labels.append(3)

    x = np.asarray(images, dtype=np.float32) / 255.0
    x = x.reshape(-1, cfg.image_size, cfg.image_size, 1)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def normalized_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert normalized grayscale image [0..1] to uint8 image [0..255]."""

    return np.clip(image * 255.0, 0, 255).astype(np.uint8)
