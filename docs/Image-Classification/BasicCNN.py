from __future__ import annotations

import sys
from pathlib import Path

import cv2
from tensorflow.keras.models import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from processing import (  # noqa: E402
    CLASS_NAMES,
    SyntheticConfig,
    TrainConfig,
    build_cnn_model,
    build_preview_samples,
    create_dataset,
    evaluate_model,
    normalized_to_uint8,
    predict_with_confidence,
    train_model,
)


def _show_samples(x_data, y_int, title_prefix: str, wait_ms: int = 500, max_samples: int = 5) -> None:
    for idx in range(min(max_samples, x_data.shape[0])):
        label = CLASS_NAMES[int(y_int[idx])]
        image = normalized_to_uint8(x_data[idx].squeeze())
        enlarged = cv2.resize(
            image,
            (image.shape[1] * 5, image.shape[0] * 5),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(f"{title_prefix} {idx + 1} ({label})", enlarged)
        cv2.waitKey(wait_ms)


def _print_epoch(epoch: int, total: int, logs: dict[str, float]) -> None:
    print(
        f"Epoch {epoch:02d}/{total:02d} | "
        f"loss={logs.get('loss', 0.0):.4f} "
        f"acc={logs.get('accuracy', 0.0):.4f} "
        f"val_loss={logs.get('val_loss', 0.0):.4f} "
        f"val_acc={logs.get('val_accuracy', 0.0):.4f}"
    )


def main() -> None:
    model_path = REPO_ROOT / "output" / "shape_classifier.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    data_cfg = SyntheticConfig(
        image_size=28,
        samples_per_class=500,
        test_size=0.2,
        apply_rotation=True,
        rotation_range=18.0,
        apply_noise=True,
        noise_std=10.0,
        seed=42,
    )
    train_cfg = TrainConfig(epochs=5, batch_size=32)

    print("\n[1] Generating synthetic shape data...")
    x_train, x_test, y_train, y_test, y_train_int, y_test_int = create_dataset(data_cfg)
    print(f"Train set: {x_train.shape[0]} images | Test set: {x_test.shape[0]} images")

    print("[2] Displaying training and test previews...")
    _show_samples(x_train, y_train_int, "Train Sample", wait_ms=450)
    _show_samples(x_test, y_test_int, "Test Sample", wait_ms=450)
    cv2.destroyAllWindows()

    print("[3] Building and training CNN...")
    model = build_cnn_model(image_size=data_cfg.image_size)
    train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        cfg=train_cfg,
        progress_callback=_print_epoch,
    )

    print(f"[4] Saving model to: {model_path}")
    model.save(model_path)

    print("[5] Loading model and evaluating...")
    model = load_model(model_path)
    metrics = evaluate_model(model, x_test, y_test)
    print(
        "Model evaluation on test set -> "
        f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy'] * 100:.2f}%"
    )

    print("[6] Running confidence predictions on fresh samples...")
    pred_x, pred_true = build_preview_samples(data_cfg, count_per_class=1, seed_offset=999)
    pred_ids, confidence = predict_with_confidence(model, pred_x)

    for idx in range(pred_x.shape[0]):
        true_label = CLASS_NAMES[int(pred_true[idx])]
        pred_label = CLASS_NAMES[int(pred_ids[idx])]
        conf = float(confidence[idx])
        correctness = "Correct" if int(pred_true[idx]) == int(pred_ids[idx]) else "Incorrect"

        print(
            f"Test image {idx + 1}: "
            f"True={true_label}, Predicted={pred_label}, Confidence={conf:.2f} -> {correctness}"
        )

        display = normalized_to_uint8(pred_x[idx].squeeze())
        enlarged = cv2.resize(
            display,
            (display.shape[1] * 6, display.shape[0] * 6),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow(f"{correctness}: {pred_label} ({conf:.2f})", enlarged)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
