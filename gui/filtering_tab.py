"""
Filtering Tab — Workshop 4
============================
Add noise (Gaussian / Salt & Pepper) then apply spatial filters
(Average / Median).  Shows Original → Noisy → Filtered with MSE values.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from processing import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_average_filter,
    apply_median_filter,
    compute_mse,
)


class FilteringTab(ttk.Frame):
    """Workshop 4 tab: noise injection + spatial filtering + MSE."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._noisy_image: np.ndarray | None = None
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None]

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        self._build_controls()
        self._build_canvas_row()
        self._build_mse_row()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        # --- Noise type ---
        ttk.Label(bar, text="Noise:", font=("Helvetica", 11, "bold")).pack(
            side=tk.LEFT, padx=(0, 4))

        self._noise_var = tk.StringVar(value="Gaussian")
        for noise_type in ("Gaussian", "Salt & Pepper"):
            rb = ttk.Radiobutton(bar, text=noise_type, variable=self._noise_var,
                                 value=noise_type)
            rb.pack(side=tk.LEFT, padx=(0, 8))

        # --- Noise parameter ---
        ttk.Label(bar, text="Strength:", font=("Helvetica", 10)).pack(
            side=tk.LEFT, padx=(12, 4))

        self._noise_strength_var = tk.DoubleVar(value=20.0)
        self._strength_lbl = ttk.Label(bar, text="20.00", font=("Helvetica", 10),
                                       width=6)
        self._strength_lbl.pack(side=tk.LEFT)

        strength_slider = ttk.Scale(bar, from_=1, to=100, orient=tk.HORIZONTAL,
                                    variable=self._noise_strength_var, length=120)
        strength_slider.pack(side=tk.LEFT, padx=(4, 12))
        self._noise_strength_var.trace_add("write", self._on_strength_changed)

        add_btn = ttk.Button(bar, text="Add Noise", command=self._on_add_noise)
        add_btn.pack(side=tk.LEFT, padx=(0, 16))

        # --- Filter type ---
        sep = ttk.Separator(bar, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(bar, text="Filter:", font=("Helvetica", 11, "bold")).pack(
            side=tk.LEFT, padx=(0, 4))

        self._filter_var = tk.StringVar(value="Average")
        for f_type in ("Average", "Median"):
            rb = ttk.Radiobutton(bar, text=f_type, variable=self._filter_var,
                                 value=f_type)
            rb.pack(side=tk.LEFT, padx=(0, 8))

        # --- Kernel size ---
        ttk.Label(bar, text="Kernel:", font=("Helvetica", 10)).pack(
            side=tk.LEFT, padx=(8, 4))

        self._ksize_var = tk.IntVar(value=5)
        for k in (3, 5, 7, 9):
            rb = ttk.Radiobutton(bar, text=str(k), variable=self._ksize_var, value=k)
            rb.pack(side=tk.LEFT, padx=(0, 4))

        apply_btn = ttk.Button(bar, text="Apply Filter", command=self._on_apply_filter)
        apply_btn.pack(side=tk.LEFT, padx=(12, 0))

    def _on_strength_changed(self, *_args) -> None:
        self._strength_lbl.config(text=f"{self._noise_strength_var.get():.2f}")

    # ------------------------------------------------------------------
    # Canvas Row
    # ------------------------------------------------------------------

    def _build_canvas_row(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.rowconfigure(1, weight=1)

        titles = ["Original", "Noisy", "Filtered"]
        self._canvases: list[tk.Canvas] = []

        for col, title in enumerate(titles):
            lbl = ttk.Label(frame, text=title, font=("Helvetica", 13, "bold"),
                            anchor="center")
            lbl.grid(row=0, column=col, sticky="ew", pady=(0, 4), padx=4)

            canvas = tk.Canvas(frame, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=col, sticky="nsew", padx=4)
            self._canvases.append(canvas)

    # ------------------------------------------------------------------
    # MSE Row
    # ------------------------------------------------------------------

    def _build_mse_row(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        self._mse_labels: list[ttk.Label] = []
        for col, text in enumerate(["", "MSE (noisy vs original): —", "MSE (filtered vs original): —"]):
            lbl = ttk.Label(frame, text=text, anchor="center",
                            font=("Helvetica", 11, "bold"), foreground="#f9e2af")
            lbl.grid(row=0, column=col, sticky="ew", padx=4, pady=4)
            self._mse_labels.append(lbl)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def on_image_loaded(self) -> None:
        self._noisy_image = None
        self._refresh_original()

    def _refresh_original(self) -> None:
        if self.app.original_image is None:
            return
        self._show_image(self.app.original_image, 0)

    def _on_add_noise(self) -> None:
        if self.app.original_image is None:
            return

        strength = self._noise_strength_var.get()
        noise_type = self._noise_var.get()

        if noise_type == "Gaussian":
            self._noisy_image = add_gaussian_noise(self.app.original_image, std=strength)
        else:
            amount = strength / 500.0
            self._noisy_image = add_salt_pepper_noise(self.app.original_image, amount=amount)

        self._show_image(self._noisy_image, 1)

        mse_val = compute_mse(self.app.original_image, self._noisy_image)
        self._mse_labels[1].config(text=f"MSE (noisy vs original): {mse_val:.2f}")

        self._canvases[2].delete("all")
        self._mse_labels[2].config(text="MSE (filtered vs original): —")

    def _on_apply_filter(self) -> None:
        if self._noisy_image is None:
            return

        ksize = self._ksize_var.get()
        filter_type = self._filter_var.get()

        if filter_type == "Average":
            filtered = apply_average_filter(self._noisy_image, ksize)
        else:
            filtered = apply_median_filter(self._noisy_image, ksize)

        self._show_image(filtered, 2)

        mse_val = compute_mse(self.app.original_image, filtered)
        self._mse_labels[2].config(text=f"MSE (filtered vs original): {mse_val:.2f}")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _show_image(self, img: np.ndarray, idx: int) -> None:
        canvas = self._canvases[idx]
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 200)
        ch = max(canvas.winfo_height(), 200)

        h, w = img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        display = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        self._photos[idx] = photo
