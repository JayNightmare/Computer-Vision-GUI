"""
Edge Detection Tab — Workshop 4
=================================
Displays five edge-detection results side by side:
Sobel, Scharr, Laplacian, Roberts, and Prewitt.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from processing import (
    edge_sobel,
    edge_scharr,
    edge_laplacian,
    edge_roberts,
    edge_prewitt,
)

_DETECTORS = [
    ("Sobel", edge_sobel),
    ("Scharr", edge_scharr),
    ("Laplacian", edge_laplacian),
    ("Roberts", edge_roberts),
    ("Prewitt", edge_prewitt),
]


class EdgeDetectionTab(ttk.Frame):
    """Workshop 4 tab: comparative edge detection."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._photos: list[ImageTk.PhotoImage | None] = [None] * len(_DETECTORS)

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        info = ttk.Label(
            self,
            text="Edge detection is applied to the grayscale version of the loaded image.",
            font=("Helvetica", 10, "italic"),
            anchor="center",
        )
        info.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 2))

        container = ttk.Frame(self)
        container.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._canvases: list[tk.Canvas] = []
        for col, (name, _fn) in enumerate(_DETECTORS):
            container.columnconfigure(col, weight=1)
            container.rowconfigure(1, weight=1)

            lbl = ttk.Label(container, text=name, font=("Helvetica", 13, "bold"),
                            anchor="center")
            lbl.grid(row=0, column=col, sticky="ew", pady=(0, 4), padx=4)

            canvas = tk.Canvas(container, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=col, sticky="nsew", padx=4)
            self._canvases.append(canvas)

    def on_image_loaded(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        for idx, (_name, fn) in enumerate(_DETECTORS):
            edges = fn(self.app.original_image)
            self._show_image(edges, idx)

    def _show_image(self, gray_img: np.ndarray, idx: int) -> None:
        canvas = self._canvases[idx]
        canvas.update_idletasks()
        cw = max(canvas.winfo_width(), 150)
        ch = max(canvas.winfo_height(), 150)

        h, w = gray_img.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
        self._photos[idx] = photo
