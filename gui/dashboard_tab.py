"""
Dashboard Tab
=============
Main editing view: original vs altered images, their histograms,
brightness / contrast / gamma sliders, and equalize / reset buttons.
"""

from __future__ import annotations

import io
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from processing import (
    adjust_brightness,
    adjust_contrast,
    compute_histogram,
    gamma_correction,
    histogram_equalization,
)

_PREVIEW_HEIGHT = 300
_HIST_COLORS = {"b": "#89b4fa", "g": "#a6e3a1", "r": "#f38ba8"}


class DashboardTab(ttk.Frame):
    """Dashboard tab with image preview, histograms, and adjustment controls."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._equalized = False
        self._grayscale = False
        self._update_job = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=2)
        self.rowconfigure(2, weight=0)

        self._build_image_row()
        self._build_histogram_row()
        self._build_controls_row()

    def _build_image_row(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        lbl_orig = ttk.Label(frame, text="Original", font=("Helvetica", 13, "bold"),
                             anchor="center")
        lbl_orig.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        lbl_alt = ttk.Label(frame, text="Altered", font=("Helvetica", 13, "bold"),
                            anchor="center")
        lbl_alt.grid(row=0, column=1, sticky="ew", pady=(0, 4))

        self._original_canvas = tk.Canvas(frame, bg="#181825", highlightthickness=0)
        self._original_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 4))

        self._altered_canvas = tk.Canvas(frame, bg="#181825", highlightthickness=0)
        self._altered_canvas.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        frame.rowconfigure(1, weight=1)

        self._orig_photo = None
        self._alt_photo = None

    def _build_histogram_row(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self._orig_hist_fig = Figure(figsize=(5, 2), dpi=80,
                                     facecolor="#1e1e2e", edgecolor="#1e1e2e")
        self._orig_hist_ax = self._orig_hist_fig.add_subplot(111)
        self._style_axis(self._orig_hist_ax, "Original Histogram")
        self._orig_hist_canvas = FigureCanvasTkAgg(self._orig_hist_fig, master=frame)
        self._orig_hist_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self._alt_hist_fig = Figure(figsize=(5, 2), dpi=80,
                                    facecolor="#1e1e2e", edgecolor="#1e1e2e")
        self._alt_hist_ax = self._alt_hist_fig.add_subplot(111)
        self._style_axis(self._alt_hist_ax, "Altered Histogram")
        self._alt_hist_canvas = FigureCanvasTkAgg(self._alt_hist_fig, master=frame)
        self._alt_hist_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        frame.rowconfigure(0, weight=1)

    def _build_controls_row(self) -> None:
        outer = ttk.LabelFrame(self, text="Adjustments")
        outer.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.columnconfigure(2, weight=1)
        outer.columnconfigure(3, weight=0)

        self.brightness_var = tk.DoubleVar(value=0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.gamma_var = tk.DoubleVar(value=1.0)

        self._make_slider(outer, 0, "Brightness", self.brightness_var, -100, 100)
        self._make_slider(outer, 1, "Contrast", self.contrast_var, 0.5, 3.0)
        self._make_slider(outer, 2, "Gamma", self.gamma_var, 0.1, 3.0)

        btn_frame = ttk.Frame(outer)
        btn_frame.grid(row=0, column=3, rowspan=2, padx=(16, 8), pady=8, sticky="ns")

        eq_btn = ttk.Button(btn_frame, text="Equalize", command=self._on_equalize)
        eq_btn.pack(fill=tk.X, pady=(0, 6))

        # Add Grayscale button
        gray_btn = ttk.Button(btn_frame, text="Grayscale", command=self._on_grayscale)
        gray_btn.pack(fill=tk.X, pady=(0, 6))

        rst_btn = ttk.Button(btn_frame, text="Reset", command=self._on_reset)
        rst_btn.pack(fill=tk.X)

    def _make_slider(self, parent: tk.Widget, col: int, label: str,
                     var: tk.DoubleVar, from_: float, to: float) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, sticky="ew", padx=8, pady=8)
        frame.columnconfigure(0, weight=1)

        value_label = ttk.Label(frame, text=f"{label}: {var.get():.2f}",
                                anchor="center", font=("Helvetica", 10))
        value_label.grid(row=0, column=0, sticky="ew")

        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL,
                           variable=var, length=200)
        slider.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        def _on_change(*_args):
            value_label.config(text=f"{label}: {var.get():.2f}")
            self._schedule_update()

        var.trace_add("write", _on_change)

    def _schedule_update(self) -> None:
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(50, self._refresh)

    def on_image_loaded(self) -> None:
        self._equalized = False
        self._grayscale = False
        self._refresh()

    def _refresh(self) -> None:
        self._update_job = None
        if self.app.original_image is None:
            return

        self._show_image(self.app.original_image, self._original_canvas, is_original=True)
        self._draw_histogram(self.app.original_image, self._orig_hist_ax,
                             self._orig_hist_canvas, "Original Histogram")

        altered = self._compute_altered()
        self._show_image(altered, self._altered_canvas, is_original=False)
        self._draw_histogram(altered, self._alt_hist_ax,
                             self._alt_hist_canvas, "Altered Histogram")

    def _compute_altered(self) -> np.ndarray:
        img = self.app.original_image.copy()

        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        gamma = self.gamma_var.get()

        if brightness != 0:
            img = adjust_brightness(img, brightness)
        if contrast != 1.0:
            img = adjust_contrast(img, contrast)
        if gamma != 1.0:
            img = gamma_correction(img, gamma)
        if self._equalized:
            img = histogram_equalization(img)
        if self._grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def _show_image(self, img: np.ndarray, canvas: tk.Canvas,
                    is_original: bool) -> None:
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 200)
        ch = max(canvas.winfo_height(), 200)

        h, w = img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if img.ndim == 2:
            display = img
        else:
            display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)

        if is_original:
            self._orig_photo = photo
        else:
            self._alt_photo = photo

    def _draw_histogram(self, img: np.ndarray, ax, canvas,
                        title: str) -> None:
        ax.clear()
        self._style_axis(ax, title)

        hist_data = compute_histogram(img)

        if "gray" in hist_data:
            ax.plot(hist_data["gray"], color="#cdd6f4", linewidth=1.2, alpha=0.85)
            ax.legend(["Gray"], fontsize=7,
                      loc="upper right", facecolor="#313244", edgecolor="#45475a",
                      labelcolor="#cdd6f4")
        else:
            for channel, color in _HIST_COLORS.items():
                ax.plot(hist_data[channel], color=color, linewidth=1.2, alpha=0.85)
            ax.legend(["Blue", "Green", "Red"], fontsize=7,
                      loc="upper right", facecolor="#313244", edgecolor="#45475a",
                      labelcolor="#cdd6f4")

        ax.set_xlim([0, 256])
        canvas.draw_idle()

    def _on_equalize(self) -> None:
        self._equalized = not self._equalized
        self._refresh()

    def _on_grayscale(self) -> None:
        self._grayscale = not self._grayscale
        self._refresh()

    def _on_reset(self) -> None:
        self._equalized = False
        self._grayscale = False
        self.brightness_var.set(0)
        self.contrast_var.set(1.0)
        self.gamma_var.set(1.0)

    @staticmethod
    def _style_axis(ax, title: str) -> None:
        ax.set_facecolor("#181825")
        ax.set_title(title, fontsize=10, color="#cdd6f4", pad=6)
        ax.tick_params(colors="#6c7086", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#45475a")
