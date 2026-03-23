"""CNN Workshop 1 tab group with dataset, training, and results subgroups."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from processing import (
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

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


class CNNTab(ttk.Frame):
    """Top-level CNN workshop tab with nested subgroups."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app

        self.x_train: np.ndarray | None = None
        self.x_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.y_train_int: np.ndarray | None = None
        self.y_test_int: np.ndarray | None = None

        self.model = None
        self.history: dict[str, list[float]] = {}
        self.metrics: dict[str, float] = {}

        self.prediction_x: np.ndarray | None = None
        self.prediction_true: np.ndarray | None = None
        self.prediction_ids: np.ndarray | None = None
        self.prediction_conf: np.ndarray | None = None

        self.image_size_var = tk.IntVar(value=28)
        self.samples_per_class_var = tk.IntVar(value=500)
        self.test_split_var = tk.DoubleVar(value=0.2)
        self.apply_rotation_var = tk.BooleanVar(value=True)
        self.rotation_range_var = tk.DoubleVar(value=18.0)
        self.apply_noise_var = tk.BooleanVar(value=True)
        self.noise_std_var = tk.DoubleVar(value=10.0)
        self.seed_var = tk.IntVar(value=42)

        self.mode_var = tk.StringVar(value="Workshop/full")
        self.epochs_var = tk.IntVar(value=10)
        self.batch_size_var = tk.IntVar(value=32)
        self.tf_log_level_var = tk.StringVar(value="ERROR")

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self, style="Inner.TNotebook")
        notebook.grid(row=0, column=0, sticky="nsew")

        self.dataset_tab = CnnDatasetTab(notebook, self)
        self.train_tab = CnnTrainTab(notebook, self)
        self.results_tab = CnnResultsTab(notebook, self)

        notebook.add(self.dataset_tab, text="  Dataset & Augmentation  ")
        notebook.add(self.train_tab, text="  Train & Evaluate  ")
        notebook.add(self.results_tab, text="  Results  ")

        self._sub_notebook = notebook

    def on_image_loaded(self) -> None:
        # CNN workflow is synthetic-data based and does not require a loaded image.
        return

    def _synthetic_config(self) -> SyntheticConfig:
        return SyntheticConfig(
            image_size=max(16, int(self.image_size_var.get())),
            samples_per_class=max(20, int(self.samples_per_class_var.get())),
            test_size=float(min(max(self.test_split_var.get(), 0.05), 0.45)),
            apply_rotation=bool(self.apply_rotation_var.get()),
            rotation_range=float(max(self.rotation_range_var.get(), 1.0)),
            apply_noise=bool(self.apply_noise_var.get()),
            noise_std=float(max(self.noise_std_var.get(), 0.0)),
            seed=int(self.seed_var.get()),
        )

    def _train_config(self) -> TrainConfig:
        return TrainConfig(
            epochs=max(1, int(self.epochs_var.get())),
            batch_size=max(8, int(self.batch_size_var.get())),
        )

    def apply_mode_defaults(self) -> None:
        mode = self.mode_var.get().strip().lower()
        if mode == "fast demo":
            self.samples_per_class_var.set(200)
            self.epochs_var.set(4)
            self.batch_size_var.set(32)
            self.rotation_range_var.set(14.0)
            self.noise_std_var.set(8.0)
        else:
            self.samples_per_class_var.set(500)
            self.epochs_var.set(10)
            self.batch_size_var.set(32)
            self.rotation_range_var.set(18.0)
            self.noise_std_var.set(10.0)

    def generate_dataset(self) -> bool:
        cfg = self._synthetic_config()
        try:
            (
                self.x_train,
                self.x_test,
                self.y_train,
                self.y_test,
                self.y_train_int,
                self.y_test_int,
            ) = create_dataset(cfg)
        except Exception as exc:  # pragma: no cover - UI guard
            messagebox.showerror("Dataset Generation Failed", str(exc))
            return False

        self.dataset_tab.refresh_preview()
        self.train_tab.show_dataset_ready()
        self.results_tab.refresh_from_state()
        return True

    def train_and_evaluate(
        self,
        progress_callback,
        log_callback,
    ) -> bool:
        if self.x_train is None or self.x_test is None or self.y_train is None or self.y_test is None:
            log_callback("No dataset found. Generating dataset first...")
            if not self.generate_dataset():
                return False

        if self.x_train is None or self.x_test is None or self.y_train is None or self.y_test is None:
            log_callback("Dataset state is incomplete.")
            return False

        cfg = self._synthetic_config()
        train_cfg = self._train_config()

        try:
            self.model = build_cnn_model(image_size=cfg.image_size)
            log_callback("Model initialized.")
            history_obj = train_model(
                model=self.model,
                x_train=self.x_train,
                y_train=self.y_train,
                x_val=self.x_test,
                y_val=self.y_test,
                cfg=train_cfg,
                progress_callback=progress_callback,
            )
            self.history = {k: list(v) for k, v in history_obj.history.items()}
            self.metrics = evaluate_model(self.model, self.x_test, self.y_test)

            self.prediction_x, self.prediction_true = build_preview_samples(cfg, count_per_class=4)
            self.prediction_ids, self.prediction_conf = predict_with_confidence(self.model, self.prediction_x)

            self.train_tab.refresh_training_views()
            self.results_tab.refresh_from_state()
            return True
        except Exception as exc:  # pragma: no cover - UI guard
            messagebox.showerror("Training Failed", str(exc))
            log_callback(f"Training failed: {exc}")
            return False

    def save_model(self) -> None:
        if self.model is None:
            messagebox.showinfo("Save Model", "Train a model before saving.")
            return

        initial_dir = str(_DEFAULT_OUTPUT_DIR) if _DEFAULT_OUTPUT_DIR.is_dir() else "."
        save_path = filedialog.asksaveasfilename(
            title="Save Keras model",
            initialdir=initial_dir,
            initialfile="shape_classifier.keras",
            defaultextension=".keras",
            filetypes=[("Keras Model", "*.keras")],
        )
        if not save_path:
            return

        self.model.save(save_path)
        messagebox.showinfo("Save Model", f"Model saved to:\n{save_path}")


class CnnDatasetTab(ttk.Frame):
    """Dataset and augmentation controls with sample previews."""

    def __init__(self, parent: tk.Widget, controller: CNNTab) -> None:
        super().__init__(parent)
        self.controller = controller
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None, None]

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self, text="Synthetic Data Controls")
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(controls, text="Image Size:").grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")
        ttk.Spinbox(controls, from_=16, to=96, increment=4, textvariable=self.controller.image_size_var, width=8).grid(
            row=0, column=1, padx=(0, 10), pady=6, sticky="w"
        )

        ttk.Label(controls, text="Samples/Class:").grid(row=0, column=2, padx=(0, 4), pady=6, sticky="w")
        ttk.Spinbox(controls, from_=20, to=3000, increment=20, textvariable=self.controller.samples_per_class_var, width=10).grid(
            row=0, column=3, padx=(0, 10), pady=6, sticky="w"
        )

        ttk.Label(controls, text="Test Split:").grid(row=0, column=4, padx=(0, 4), pady=6, sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.05,
            to=0.45,
            increment=0.05,
            textvariable=self.controller.test_split_var,
            width=8,
            format="%.2f",
        ).grid(row=0, column=5, padx=(0, 12), pady=6, sticky="w")

        ttk.Checkbutton(controls, text="Apply Rotation", variable=self.controller.apply_rotation_var).grid(
            row=1, column=0, padx=(8, 4), pady=6, sticky="w"
        )
        ttk.Label(controls, text="Range:").grid(row=1, column=1, padx=(0, 4), pady=6, sticky="e")
        ttk.Scale(
            controls,
            from_=5,
            to=40,
            variable=self.controller.rotation_range_var,
            orient=tk.HORIZONTAL,
            length=140,
        ).grid(row=1, column=2, padx=(0, 10), pady=6, sticky="w")

        ttk.Checkbutton(controls, text="Apply Noise", variable=self.controller.apply_noise_var).grid(
            row=1, column=3, padx=(0, 4), pady=6, sticky="w"
        )
        ttk.Label(controls, text="Std:").grid(row=1, column=4, padx=(0, 4), pady=6, sticky="e")
        ttk.Scale(
            controls,
            from_=0,
            to=30,
            variable=self.controller.noise_std_var,
            orient=tk.HORIZONTAL,
            length=140,
        ).grid(row=1, column=5, padx=(0, 10), pady=6, sticky="w")

        ttk.Label(controls, text="Seed:").grid(row=1, column=6, padx=(0, 4), pady=6, sticky="e")
        ttk.Spinbox(controls, from_=0, to=9999, textvariable=self.controller.seed_var, width=8).grid(
            row=1, column=7, padx=(0, 8), pady=6, sticky="w"
        )

        ttk.Button(controls, text="Generate Dataset", command=self._on_generate).grid(
            row=0, column=7, padx=(8, 8), pady=6, sticky="e"
        )

        self._status = ttk.Label(
            controls,
            text="No dataset generated yet.",
            foreground="#f9e2af",
            font=("Helvetica", 10, "bold"),
        )
        self._status.grid(row=2, column=0, columnspan=8, padx=8, pady=(2, 8), sticky="w")

        preview = ttk.Frame(self)
        preview.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        titles = ["Sample 1", "Sample 2", "Sample 3", "Sample 4"]
        self._canvases: list[tk.Canvas] = []
        for idx, title in enumerate(titles):
            preview.columnconfigure(idx, weight=1)
            ttk.Label(preview, text=title, font=("Helvetica", 11, "bold")).grid(
                row=0, column=idx, sticky="ew", padx=4, pady=(0, 4)
            )
            canvas = tk.Canvas(preview, bg="#181825", highlightthickness=0, height=220)
            canvas.grid(row=1, column=idx, sticky="nsew", padx=4)
            self._canvases.append(canvas)

    def _on_generate(self) -> None:
        if self.controller.generate_dataset():
            if self.controller.x_train is None or self.controller.x_test is None:
                return
            train_count = int(self.controller.x_train.shape[0])
            test_count = int(self.controller.x_test.shape[0])
            self._status.config(
                text=(
                    f"Dataset generated: train={train_count} test={test_count} "
                    f"(size={self.controller.image_size_var.get()}x{self.controller.image_size_var.get()})"
                ),
                foreground="#a6e3a1",
            )

    def refresh_preview(self) -> None:
        if self.controller.x_train is None or self.controller.y_train_int is None:
            return

        indices = [0, 1, 2, 3]
        for canvas_idx, sample_idx in enumerate(indices):
            if sample_idx >= self.controller.x_train.shape[0]:
                continue

            image = normalized_to_uint8(self.controller.x_train[sample_idx].squeeze())
            label = CLASS_NAMES[int(self.controller.y_train_int[sample_idx])]
            self._show_image(self._canvases[canvas_idx], image, canvas_idx, label)

    def _show_image(
        self,
        canvas: tk.Canvas,
        gray_image: np.ndarray,
        photo_idx: int,
        label: str,
    ) -> None:
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 160)
        ch = max(canvas.winfo_height(), 160)

        rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(rgb, (160, 160), interpolation=cv2.INTER_NEAREST)
        cv2.putText(rgb, label, (8, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 255, 170), 2, cv2.LINE_AA)

        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        self._photos[photo_idx] = photo


class CnnTrainTab(ttk.Frame):
    """Train/evaluate subgroup with metrics, logs, and confidence view."""

    def __init__(self, parent: tk.Widget, controller: CNNTab) -> None:
        super().__init__(parent)
        self.controller = controller

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self, text="Training Controls")
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(controls, text="Mode:").grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.controller.mode_var,
            values=["Fast demo", "Workshop/full"],
            state="readonly",
            width=14,
        )
        mode_combo.grid(row=0, column=1, padx=(0, 8), pady=6, sticky="w")

        ttk.Button(controls, text="Apply Mode Defaults", command=self._on_apply_mode).grid(
            row=0, column=2, padx=(0, 12), pady=6, sticky="w"
        )

        ttk.Label(controls, text="Epochs:").grid(row=0, column=3, padx=(0, 4), pady=6, sticky="w")
        ttk.Spinbox(controls, from_=1, to=100, textvariable=self.controller.epochs_var, width=8).grid(
            row=0, column=4, padx=(0, 10), pady=6, sticky="w"
        )

        ttk.Label(controls, text="Batch:").grid(row=0, column=5, padx=(0, 4), pady=6, sticky="w")
        ttk.Spinbox(controls, from_=8, to=256, increment=8, textvariable=self.controller.batch_size_var, width=8).grid(
            row=0, column=6, padx=(0, 10), pady=6, sticky="w"
        )

        ttk.Label(controls, text="TF Logs:").grid(row=0, column=7, padx=(0, 4), pady=6, sticky="w")
        tf_log_combo = ttk.Combobox(
            controls,
            textvariable=self.controller.tf_log_level_var,
            values=["ERROR", "WARNING", "INFO", "DEBUG"],
            state="readonly",
            width=9,
        )
        tf_log_combo.grid(row=0, column=8, padx=(0, 6), pady=6, sticky="w")

        ttk.Button(controls, text="Apply Logs", command=self._on_apply_tf_logs).grid(
            row=0, column=9, padx=(0, 12), pady=6, sticky="w"
        )

        ttk.Button(controls, text="Train + Evaluate", command=self._on_train).grid(
            row=0, column=10, padx=(0, 8), pady=6, sticky="w"
        )
        ttk.Button(controls, text="Save Model", command=self.controller.save_model).grid(
            row=0, column=11, padx=(0, 8), pady=6, sticky="w"
        )

        self._progress_label = ttk.Label(
            controls,
            text="Idle",
            foreground="#f9e2af",
            font=("Helvetica", 10, "bold"),
        )
        self._progress_label.grid(row=1, column=0, columnspan=12, padx=8, pady=(0, 8), sticky="w")

        self._curve_fig = Figure(figsize=(6, 3.5), dpi=90, facecolor="#1e1e2e")
        self._curve_ax = self._curve_fig.add_subplot(111)
        self._style_axis(self._curve_ax, "Training Curves")
        self._curve_canvas = FigureCanvasTkAgg(self._curve_fig, master=self)
        self._curve_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))

        side_panel = ttk.Frame(self)
        side_panel.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(2, weight=1)

        self._metric_label = ttk.Label(
            side_panel,
            text="Accuracy: -\nLoss: -",
            justify=tk.LEFT,
            font=("Helvetica", 11, "bold"),
            foreground="#a6e3a1",
        )
        self._metric_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ttk.Label(side_panel, text="Confidence Preview", font=("Helvetica", 11, "bold")).grid(
            row=1, column=0, sticky="ew", pady=(0, 4)
        )

        columns = ("idx", "true", "pred", "conf")
        self._confidence_table = ttk.Treeview(side_panel, columns=columns, show="headings", height=7)
        self._confidence_table.heading("idx", text="#")
        self._confidence_table.heading("true", text="True")
        self._confidence_table.heading("pred", text="Pred")
        self._confidence_table.heading("conf", text="Confidence")
        self._confidence_table.column("idx", width=40, anchor="center")
        self._confidence_table.column("true", width=80, anchor="center")
        self._confidence_table.column("pred", width=80, anchor="center")
        self._confidence_table.column("conf", width=90, anchor="center")
        self._confidence_table.grid(row=2, column=0, sticky="nsew")

        ttk.Label(side_panel, text="Training Log", font=("Helvetica", 11, "bold")).grid(
            row=3, column=0, sticky="ew", pady=(8, 4)
        )

        self._log = tk.Text(side_panel, height=9, bg="#181825", fg="#cdd6f4", insertbackground="#cdd6f4")
        self._log.grid(row=4, column=0, sticky="ew")

    def _on_apply_mode(self) -> None:
        self.controller.apply_mode_defaults()
        self._append_log(
            f"Applied mode '{self.controller.mode_var.get()}': "
            f"samples/class={self.controller.samples_per_class_var.get()}, "
            f"epochs={self.controller.epochs_var.get()}, "
            f"batch={self.controller.batch_size_var.get()}."
        )

    def _on_apply_tf_logs(self) -> None:
        selected = self.controller.tf_log_level_var.get().strip().upper()
        set_tf_logging_level(selected)
        self._append_log(f"TensorFlow log level set to {selected}.")

    def _on_train(self) -> None:
        self._append_log("Starting train/evaluate workflow...")
        self._progress_label.config(text="Training started...", foreground="#f9e2af")
        self.update_idletasks()

        ok = self.controller.train_and_evaluate(self._on_epoch_progress, self._append_log)
        if not ok:
            self._progress_label.config(text="Training failed.", foreground="#f38ba8")
            return

        self._progress_label.config(text="Training complete.", foreground="#a6e3a1")
        self._append_log("Training complete.")

    def _on_epoch_progress(self, epoch: int, total: int, logs: dict[str, float]) -> None:
        acc = float(logs.get("accuracy", 0.0))
        val_acc = float(logs.get("val_accuracy", 0.0))
        loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))

        self._progress_label.config(
            text=(
                f"Epoch {epoch}/{total} "
                f"loss={loss:.4f} acc={acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
        )
        self._append_log(
            f"Epoch {epoch:02d}/{total:02d} | "
            f"loss={loss:.4f}, acc={acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        self.update_idletasks()

    def _append_log(self, message: str) -> None:
        self._log.insert(tk.END, message + "\n")
        self._log.see(tk.END)

    def show_dataset_ready(self) -> None:
        if self.controller.x_train is None or self.controller.x_test is None:
            return
        self._append_log(
            f"Dataset ready: train={self.controller.x_train.shape[0]}, test={self.controller.x_test.shape[0]}"
        )

    def refresh_training_views(self) -> None:
        metrics = self.controller.metrics
        if metrics:
            self._metric_label.config(
                text=f"Accuracy: {metrics['accuracy'] * 100:.2f}%\nLoss: {metrics['loss']:.4f}"
            )

        self._refresh_curve()
        self._refresh_confidence_table()

    def _refresh_curve(self) -> None:
        history = self.controller.history

        self._curve_ax.clear()
        self._style_axis(self._curve_ax, "Training Curves")

        if not history:
            self._curve_canvas.draw_idle()
            return

        epochs = np.arange(1, len(history.get("loss", [])) + 1)
        if len(epochs) > 0:
            self._curve_ax.plot(epochs, history.get("loss", []), color="#f38ba8", linewidth=2, label="Train Loss")
            self._curve_ax.plot(epochs, history.get("val_loss", []), color="#fab387", linewidth=2, label="Val Loss")
            self._curve_ax.plot(epochs, history.get("accuracy", []), color="#89b4fa", linewidth=2, label="Train Acc")
            self._curve_ax.plot(epochs, history.get("val_accuracy", []), color="#a6e3a1", linewidth=2, label="Val Acc")
            self._curve_ax.set_xlim(1, max(int(epochs[-1]), 1))
            self._curve_ax.legend(
                fontsize=8,
                facecolor="#313244",
                edgecolor="#45475a",
                labelcolor="#cdd6f4",
                loc="best",
            )

        self._curve_canvas.draw_idle()

    def _refresh_confidence_table(self) -> None:
        for item in self._confidence_table.get_children():
            self._confidence_table.delete(item)

        if (
            self.controller.prediction_true is None
            or self.controller.prediction_ids is None
            or self.controller.prediction_conf is None
        ):
            return

        for idx, (true_id, pred_id, conf) in enumerate(
            zip(
                self.controller.prediction_true,
                self.controller.prediction_ids,
                self.controller.prediction_conf,
            ),
            start=1,
        ):
            self._confidence_table.insert(
                "",
                tk.END,
                values=(idx, CLASS_NAMES[int(true_id)], CLASS_NAMES[int(pred_id)], f"{float(conf):.2f}"),
            )

    @staticmethod
    def _style_axis(ax, title: str) -> None:
        ax.set_facecolor("#181825")
        ax.set_title(title, fontsize=11, color="#cdd6f4", pad=6)
        ax.tick_params(colors="#6c7086", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#45475a")


class CnnResultsTab(ttk.Frame):
    """Visual results subgroup for predictions and confidence outcome."""

    def __init__(self, parent: tk.Widget, controller: CNNTab) -> None:
        super().__init__(parent)
        self.controller = controller
        self._photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Button(top, text="Refresh Results", command=self.refresh_from_state).pack(side=tk.LEFT)

        self._summary = ttk.Label(
            top,
            text="Run training to generate predictions.",
            font=("Helvetica", 10, "bold"),
            foreground="#f9e2af",
        )
        self._summary.pack(side=tk.LEFT, padx=(10, 0))

        self._canvas = tk.Canvas(self, bg="#181825", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

    def refresh_from_state(self) -> None:
        if (
            self.controller.prediction_x is None
            or self.controller.prediction_true is None
            or self.controller.prediction_ids is None
            or self.controller.prediction_conf is None
        ):
            self._summary.config(text="No prediction results yet.", foreground="#f9e2af")
            self._canvas.delete("all")
            return

        montage = _build_prediction_montage(
            self.controller.prediction_x,
            self.controller.prediction_true,
            self.controller.prediction_ids,
            self.controller.prediction_conf,
        )

        correct = int(np.sum(self.controller.prediction_true == self.controller.prediction_ids))
        total = int(self.controller.prediction_true.shape[0])
        accuracy = (correct / total) * 100.0 if total else 0.0
        self._summary.config(
            text=f"Prediction preview accuracy: {accuracy:.1f}% ({correct}/{total})",
            foreground="#a6e3a1",
        )

        self._show_montage(montage)

    def _show_montage(self, bgr_image: np.ndarray) -> None:
        self._canvas.update_idletasks()
        cw = max(self._canvas.winfo_width(), 300)
        ch = max(self._canvas.winfo_height(), 240)

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        resized = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        photo = ImageTk.PhotoImage(Image.fromarray(resized))
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        self._photo = photo


def _build_prediction_montage(
    x: np.ndarray,
    true_ids: np.ndarray,
    pred_ids: np.ndarray,
    conf: np.ndarray,
    tile_size: int = 120,
    cols: int = 4,
) -> np.ndarray:
    """Create a compact tiled visualization with label/confidence overlays."""

    total = x.shape[0]
    rows = int(np.ceil(total / cols))
    out_h = rows * tile_size
    out_w = cols * tile_size
    canvas = np.full((out_h, out_w, 3), 24, dtype=np.uint8)

    for idx in range(total):
        row = idx // cols
        col = idx % cols
        y0 = row * tile_size
        x0 = col * tile_size

        gray = normalized_to_uint8(x[idx].squeeze())
        tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

        is_correct = int(true_ids[idx]) == int(pred_ids[idx])
        color = (80, 220, 140) if is_correct else (220, 120, 120)

        cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), color, 2)
        cv2.putText(
            tile,
            f"T:{CLASS_NAMES[int(true_ids[idx])][:2]} P:{CLASS_NAMES[int(pred_ids[idx])][:2]}",
            (6, tile_size - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"C:{float(conf[idx]):.2f}",
            (6, tile_size - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

    return canvas
