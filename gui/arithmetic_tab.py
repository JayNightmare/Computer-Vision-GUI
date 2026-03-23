"""
Arithmetic Tab — Workshop 4
============================
Load a second image and perform pixel-wise addition, subtraction,
or multiplication against the primary loaded image.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from processing import image_add, image_subtract, image_multiply

_DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "image"


class ArithmeticTab(ttk.Frame):
    """Workshop 4 tab: pixel-wise image arithmetic."""

    _OPERATIONS = {
        "Add": image_add,
        "Subtract": image_subtract,
        "Multiply": image_multiply,
    }

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._image_b: np.ndarray | None = None
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None]

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_controls()
        self._build_canvas_row()

    def _build_controls(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        load_btn = ttk.Button(bar, text="Load Image B…", command=self._load_image_b)
        load_btn.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(bar, text="Operation:", font=("Helvetica", 11)).pack(side=tk.LEFT)

        self._op_var = tk.StringVar(value="Add")
        op_menu = ttk.OptionMenu(bar, self._op_var, "Add", *self._OPERATIONS.keys(),
                                 command=lambda _: self._refresh())
        op_menu.pack(side=tk.LEFT, padx=(4, 0))

    def _build_canvas_row(self) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)
        frame.rowconfigure(1, weight=1)

        titles = ["Image A (loaded)", "Image B", "Result"]
        self._canvases: list[tk.Canvas] = []

        for col, title in enumerate(titles):
            lbl = ttk.Label(frame, text=title, font=("Helvetica", 13, "bold"),
                            anchor="center")
            lbl.grid(row=0, column=col, sticky="ew", pady=(0, 4), padx=4)

            canvas = tk.Canvas(frame, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=col, sticky="nsew", padx=4)
            self._canvases.append(canvas)

    def _load_image_b(self) -> None:
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("All files", "*.*"),
        ]
        initial_dir = str(_DEFAULT_IMAGE_DIR) if _DEFAULT_IMAGE_DIR.is_dir() else "."
        path = filedialog.askopenfilename(
            title="Select Image B",
            initialdir=initial_dir,
            filetypes=filetypes,
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image:\n{path}")
            return

        if self.app.original_image is not None:
            h, w = self.app.original_image.shape[:2]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        self._image_b = img
        self._refresh()

    def on_image_loaded(self) -> None:
        if self._image_b is not None and self.app.original_image is not None:
            h, w = self.app.original_image.shape[:2]
            self._image_b = cv2.resize(self._image_b, (w, h),
                                       interpolation=cv2.INTER_AREA)
        self._refresh()

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        self._show_image(self.app.original_image, 0)

        if self._image_b is not None:
            self._show_image(self._image_b, 1)

            op_fn = self._OPERATIONS[self._op_var.get()]
            result = op_fn(self.app.original_image, self._image_b)
            self._show_image(result, 2)

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
