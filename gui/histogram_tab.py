"""
Histogram Tab
=============
Shows three side-by-side views: Current (with adjustments), CLAHE,
and Negative — each with its corresponding histogram below.
"""

from __future__ import annotations

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
    apply_clahe,
    compute_histogram,
    gamma_correction,
    negative_transform,
)

_HIST_COLORS = {"b": "#89b4fa", "g": "#a6e3a1", "r": "#f38ba8"}


class HistogramTab(ttk.Frame):
    """Histogram comparison tab: Current | CLAHE | Negative."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None]
        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=2)

        titles = ["Current", "CLAHE", "Negative"]
        self._canvases: list[tk.Canvas] = []
        self._hist_figs: list[Figure] = []
        self._hist_axes = []
        self._hist_canvases: list[FigureCanvasTkAgg] = []

        for col, title in enumerate(titles):
            lbl = ttk.Label(self, text=title, font=("Helvetica", 13, "bold"),
                            anchor="center")
            lbl.grid(row=0, column=col, sticky="ew", pady=(8, 4), padx=4)

            canvas = tk.Canvas(self, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=col, sticky="nsew", padx=4, pady=(0, 4))
            self._canvases.append(canvas)

            fig = Figure(figsize=(4, 2), dpi=80,
                         facecolor="#1e1e2e", edgecolor="#1e1e2e")
            ax = fig.add_subplot(111)
            self._style_axis(ax, f"{title} Histogram")
            fc = FigureCanvasTkAgg(fig, master=self)
            fc.get_tk_widget().grid(row=2, column=col, sticky="nsew", padx=4, pady=(0, 8))

            self._hist_figs.append(fig)
            self._hist_axes.append(ax)
            self._hist_canvases.append(fc)

    def on_image_loaded(self) -> None:
        self._refresh()

    def _get_current_image(self) -> np.ndarray:
        """Apply the dashboard slider values to the original image."""
        img = self.app.original_image.copy()
        dashboard = self.app.dashboard_tab

        brightness = dashboard.brightness_var.get()
        contrast = dashboard.contrast_var.get()
        gamma = dashboard.gamma_var.get()

        if brightness != 0:
            img = adjust_brightness(img, brightness)
        if contrast != 1.0:
            img = adjust_contrast(img, contrast)
        if gamma != 1.0:
            img = gamma_correction(img, gamma)

        return img

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        current = self._get_current_image()
        clahe_img = apply_clahe(current)
        negative_img = negative_transform(current)

        images = [current, clahe_img, negative_img]
        titles = ["Current Histogram", "CLAHE Histogram", "Negative Histogram"]

        for idx, (img, title) in enumerate(zip(images, titles)):
            self._show_image(img, self._canvases[idx], idx)
            self._draw_histogram(img, self._hist_axes[idx],
                                 self._hist_canvases[idx], title)

    def _show_image(self, bgr_img: np.ndarray, canvas: tk.Canvas,
                    photo_idx: int) -> None:
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 200)
        ch = max(canvas.winfo_height(), 200)

        h, w = bgr_img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        self._photos[photo_idx] = photo

    def _draw_histogram(self, bgr_img: np.ndarray, ax, canvas,
                        title: str) -> None:
        ax.clear()
        self._style_axis(ax, title)

        hist_data = compute_histogram(bgr_img)
        for channel, color in _HIST_COLORS.items():
            ax.plot(hist_data[channel], color=color, linewidth=1.2, alpha=0.85)
        ax.set_xlim([0, 256])
        ax.legend(["Blue", "Green", "Red"], fontsize=7,
                  loc="upper right", facecolor="#313244", edgecolor="#45475a",
                  labelcolor="#cdd6f4")
        canvas.draw_idle()

    @staticmethod
    def _style_axis(ax, title: str) -> None:
        ax.set_facecolor("#181825")
        ax.set_title(title, fontsize=10, color="#cdd6f4", pad=6)
        ax.tick_params(colors="#6c7086", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#45475a")
