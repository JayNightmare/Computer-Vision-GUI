"""
Workshop 5 - Image Segmentation
===============================
Includes:
- Thresholding methods (Adaptive + Otsu)
- Assignment 1 - Task 2 (Segmentation + Object Detection)
- Assignment 2 - Object Detection
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from processing import (
    adaptive_threshold,
    connected_components_analysis,
    contour_object_detection,
    otsu_threshold,
)

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


class ThresholdingMethodsTab(ttk.Frame):
    """Task 1 tab: Adaptive thresholding and Otsu's method."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None, None]
        self._last_outputs: dict[str, np.ndarray] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(controls, text="Adaptive Method:", font=("Helvetica", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )

        self._adaptive_method_var = tk.StringVar(value="Gaussian")
        adaptive_combo = ttk.Combobox(
            controls,
            textvariable=self._adaptive_method_var,
            values=["Gaussian", "Mean"],
            state="readonly",
            width=10,
        )
        adaptive_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(controls, text="Block Size:").pack(side=tk.LEFT, padx=(0, 4))
        self._block_var = tk.IntVar(value=11)
        self._block_label = ttk.Label(controls, text="11", width=3)
        self._block_label.pack(side=tk.LEFT)

        block_scale = ttk.Scale(
            controls,
            from_=3,
            to=41,
            orient=tk.HORIZONTAL,
            variable=self._block_var,
            length=120,
        )
        block_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._block_var.trace_add("write", self._on_block_changed)

        ttk.Label(controls, text="C:").pack(side=tk.LEFT, padx=(0, 4))
        self._c_var = tk.DoubleVar(value=2.0)
        self._c_label = ttk.Label(controls, text="2.0", width=4)
        self._c_label.pack(side=tk.LEFT)

        c_scale = ttk.Scale(
            controls,
            from_=-10,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self._c_var,
            length=120,
        )
        c_scale.pack(side=tk.LEFT, padx=(4, 12))
        self._c_var.trace_add("write", self._on_c_changed)

        ttk.Label(controls, text="Otsu Blur:").pack(side=tk.LEFT, padx=(0, 4))
        self._blur_var = tk.IntVar(value=5)
        for val in (0, 3, 5, 7):
            ttk.Radiobutton(
                controls,
                text=("None" if val == 0 else str(val)),
                variable=self._blur_var,
                value=val,
            ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(controls, text="Run", command=self._refresh).pack(side=tk.LEFT, padx=(10, 6))
        ttk.Button(controls, text="Save Result", command=self._on_save_result).pack(side=tk.LEFT)

        self._status_label = ttk.Label(
            controls,
            text="Otsu thresholds: raw -, blur -",
            foreground="#f9e2af",
            font=("Helvetica", 10, "bold"),
        )
        self._status_label.pack(side=tk.RIGHT)

        grid = ttk.Frame(self)
        grid.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        titles = ["Original (Gray)", "Adaptive", "Otsu (Raw)", "Otsu (+Blur)"]
        self._canvases: list[tk.Canvas] = []

        for idx, title in enumerate(titles):
            row = idx // 2
            col = idx % 2
            grid.columnconfigure(col, weight=1)
            grid.rowconfigure(2 * row + 1, weight=1)

            ttk.Label(grid, text=title, font=("Helvetica", 12, "bold")).grid(
                row=2 * row,
                column=col,
                sticky="ew",
                padx=4,
                pady=(0, 4),
            )

            canvas = tk.Canvas(grid, bg="#181825", highlightthickness=0)
            canvas.grid(row=2 * row + 1, column=col, sticky="nsew", padx=4, pady=(0, 6))
            self._canvases.append(canvas)

    def _on_block_changed(self, *_args) -> None:
        block_size = _ensure_odd(int(round(self._block_var.get())))
        if self._block_var.get() != block_size:
            self._block_var.set(block_size)
        self._block_label.config(text=str(block_size))

    def _on_c_changed(self, *_args) -> None:
        self._c_label.config(text=f"{self._c_var.get():.1f}")

    def on_image_loaded(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        gray = _to_gray(self.app.original_image)

        method = self._adaptive_method_var.get().lower()
        block_size = _ensure_odd(int(round(self._block_var.get())))
        c_value = float(self._c_var.get())

        adaptive_bin = adaptive_threshold(
            gray,
            method=method,
            block_size=block_size,
            c_value=c_value,
        )

        raw_t, otsu_raw = otsu_threshold(gray, blur_ksize=0)
        blur_t, otsu_blur = otsu_threshold(gray, blur_ksize=int(self._blur_var.get()))

        self._status_label.config(text=f"Otsu thresholds: raw {raw_t:.2f}, blur {blur_t:.2f}")

        self._show_image(gray, 0)
        self._show_image(adaptive_bin, 1)
        self._show_image(otsu_raw, 2)
        self._show_image(otsu_blur, 3)

        self._last_outputs = {
            "adaptive": adaptive_bin,
            "otsu_raw": otsu_raw,
            "otsu_blur": otsu_blur,
        }

    def _on_save_result(self) -> None:
        if not self._last_outputs:
            messagebox.showinfo("Save Result", "No result available to save yet.")
            return

        image = self._last_outputs.get("otsu_blur")
        if image is None:
            image = self._last_outputs.get("adaptive")
        if image is None:
            messagebox.showinfo("Save Result", "No result available to save yet.")
            return

        _save_image_dialog(image, "thresholding_result.png")

    def _show_image(self, image: np.ndarray, idx: int) -> None:
        _show_image_on_canvas(self._canvases[idx], image, self._photos, idx)


class Assignment1Task2Tab(ttk.Frame):
    """Assignment 1 - Task 2 tab with nested Segmentation and Object Detection tabs."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self, style="Inner.TNotebook")
        notebook.grid(row=0, column=0, sticky="nsew")

        self.segmentation_tab = A1SegmentationSubTab(notebook, app)
        self.object_detection_tab = A1ObjectDetectionSubTab(notebook, app)

        notebook.add(self.segmentation_tab, text="  Segmentation  ")
        notebook.add(self.object_detection_tab, text="  Object Detection  ")

        self._sub_notebook = notebook
        self._sub_notebook.bind("<<NotebookTabChanged>>", self._on_sub_tab_changed)

    def _on_sub_tab_changed(self, _event: tk.Event) -> None:
        if self.app.original_image is None:
            return

        selected = self._sub_notebook.select()
        if not selected:
            return

        widget = self.nametowidget(selected)
        if hasattr(widget, "on_image_loaded"):
            widget.on_image_loaded()

    def on_image_loaded(self) -> None:
        self.segmentation_tab.on_image_loaded()
        self.object_detection_tab.on_image_loaded()


class A1SegmentationSubTab(ttk.Frame):
    """Assignment 1 - Task 2 (Segmentation)."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None]
        self._last_binary: np.ndarray | None = None
        self._last_labeled: np.ndarray | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(controls, text="Method:", font=("Helvetica", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._method_var = tk.StringVar(value="Otsu")
        method_combo = ttk.Combobox(
            controls,
            textvariable=self._method_var,
            values=["Otsu", "Adaptive Gaussian", "Adaptive Mean"],
            state="readonly",
            width=18,
        )
        method_combo.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(controls, text="Block:").pack(side=tk.LEFT, padx=(0, 4))
        self._block_var = tk.IntVar(value=11)
        self._block_lbl = ttk.Label(controls, text="11", width=3)
        self._block_lbl.pack(side=tk.LEFT)

        block_scale = ttk.Scale(
            controls,
            from_=3,
            to=41,
            orient=tk.HORIZONTAL,
            variable=self._block_var,
            length=110,
        )
        block_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._block_var.trace_add("write", self._on_block_changed)

        ttk.Label(controls, text="C:").pack(side=tk.LEFT, padx=(0, 4))
        self._c_var = tk.DoubleVar(value=2.0)
        self._c_lbl = ttk.Label(controls, text="2.0", width=4)
        self._c_lbl.pack(side=tk.LEFT)

        c_scale = ttk.Scale(
            controls,
            from_=-10,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self._c_var,
            length=110,
        )
        c_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._c_var.trace_add("write", self._on_c_changed)

        ttk.Button(controls, text="Segment", command=self._refresh).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Save Labeled", command=self._on_save_labeled).pack(side=tk.LEFT)

        self._objects_lbl = ttk.Label(
            controls,
            text="Objects detected: -",
            font=("Helvetica", 10, "bold"),
            foreground="#a6e3a1",
        )
        self._objects_lbl.pack(side=tk.RIGHT)

        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        titles = ["Original (Gray)", "Binary Mask", "Labeled Regions"]
        self._canvases: list[tk.Canvas] = []
        for idx, title in enumerate(titles):
            frame.columnconfigure(idx, weight=1)
            frame.rowconfigure(1, weight=1)

            ttk.Label(frame, text=title, font=("Helvetica", 12, "bold")).grid(
                row=0,
                column=idx,
                sticky="ew",
                padx=4,
                pady=(0, 4),
            )

            canvas = tk.Canvas(frame, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=idx, sticky="nsew", padx=4)
            self._canvases.append(canvas)

    def _on_block_changed(self, *_args) -> None:
        block_size = _ensure_odd(int(round(self._block_var.get())))
        if self._block_var.get() != block_size:
            self._block_var.set(block_size)
        self._block_lbl.config(text=str(block_size))

    def _on_c_changed(self, *_args) -> None:
        self._c_lbl.config(text=f"{self._c_var.get():.1f}")

    def on_image_loaded(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        gray = _to_gray(self.app.original_image)
        method = self._method_var.get()

        if method == "Otsu":
            _, binary = otsu_threshold(gray, blur_ksize=0)
        elif method == "Adaptive Gaussian":
            binary = adaptive_threshold(
                gray,
                method="gaussian",
                block_size=int(self._block_var.get()),
                c_value=float(self._c_var.get()),
            )
        else:
            binary = adaptive_threshold(
                gray,
                method="mean",
                block_size=int(self._block_var.get()),
                c_value=float(self._c_var.get()),
            )

        colored_labels, _, measurements = connected_components_analysis(binary)

        self._objects_lbl.config(text=f"Objects detected: {len(measurements)}")

        self._show_image(gray, 0)
        self._show_image(binary, 1)
        self._show_image(colored_labels, 2)

        self._last_binary = binary
        self._last_labeled = colored_labels

    def _on_save_labeled(self) -> None:
        if self._last_labeled is None:
            messagebox.showinfo("Save Labeled", "Run segmentation first.")
            return
        _save_image_dialog(self._last_labeled, "a1_task2_labeled.png")

    def _show_image(self, image: np.ndarray, idx: int) -> None:
        _show_image_on_canvas(self._canvases[idx], image, self._photos, idx)


class A1ObjectDetectionSubTab(ttk.Frame):
    """Assignment 1 - Task 2 (Object Detection from segmented mask)."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None]

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ttk.Label(controls, text="Mask Source:", font=("Helvetica", 10, "bold")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self._mask_var = tk.StringVar(value="Otsu")
        mask_combo = ttk.Combobox(
            controls,
            textvariable=self._mask_var,
            values=["Otsu", "Adaptive Gaussian", "Adaptive Mean"],
            state="readonly",
            width=18,
        )
        mask_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(controls, text="Min Area:").pack(side=tk.LEFT, padx=(0, 4))
        self._min_area_var = tk.DoubleVar(value=50.0)
        self._min_area_lbl = ttk.Label(controls, text="50", width=5)
        self._min_area_lbl.pack(side=tk.LEFT)

        area_scale = ttk.Scale(
            controls,
            from_=1,
            to=2000,
            orient=tk.HORIZONTAL,
            variable=self._min_area_var,
            length=140,
        )
        area_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._min_area_var.trace_add("write", self._on_area_changed)

        ttk.Label(controls, text="Morph Open:").pack(side=tk.LEFT, padx=(0, 4))
        self._morph_open_var = tk.IntVar(value=3)
        self._morph_open_lbl = ttk.Label(controls, text="3", width=3)
        self._morph_open_lbl.pack(side=tk.LEFT)
        open_scale = ttk.Scale(
            controls,
            from_=1,
            to=11,
            orient=tk.HORIZONTAL,
            variable=self._morph_open_var,
            length=100,
        )
        open_scale.pack(side=tk.LEFT, padx=(4, 8))
        self._morph_open_var.trace_add("write", self._on_morph_open_changed)

        ttk.Label(controls, text="Morph Close:").pack(side=tk.LEFT, padx=(0, 4))
        self._morph_close_var = tk.IntVar(value=3)
        self._morph_close_lbl = ttk.Label(controls, text="3", width=3)
        self._morph_close_lbl.pack(side=tk.LEFT)
        close_scale = ttk.Scale(
            controls,
            from_=1,
            to=11,
            orient=tk.HORIZONTAL,
            variable=self._morph_close_var,
            length=100,
        )
        close_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._morph_close_var.trace_add("write", self._on_morph_close_changed)

        ttk.Button(controls, text="Detect Objects", command=self._refresh).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controls, text="Save Annotated", command=self._on_save_annotated).pack(side=tk.LEFT)

        self._summary_label = ttk.Label(
            controls,
            text="Detected objects: -",
            font=("Helvetica", 10, "bold"),
            foreground="#a6e3a1",
        )
        self._summary_label.pack(side=tk.RIGHT)

        image_frame = ttk.Frame(self)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))

        titles = ["Original (Gray)", "Binary Mask", "Annotated"]
        self._canvases: list[tk.Canvas] = []
        for idx, title in enumerate(titles):
            image_frame.columnconfigure(idx, weight=1)
            image_frame.rowconfigure(1, weight=1)

            ttk.Label(image_frame, text=title, font=("Helvetica", 12, "bold")).grid(
                row=0,
                column=idx,
                sticky="ew",
                padx=4,
                pady=(0, 4),
            )
            canvas = tk.Canvas(image_frame, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=idx, sticky="nsew", padx=4)
            self._canvases.append(canvas)

        table_frame = ttk.Frame(self)
        table_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        tools = ttk.Frame(table_frame)
        tools.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(tools, text="Sort By:").pack(side=tk.LEFT, padx=(0, 4))
        self._sort_var = tk.StringVar(value="Label")
        ttk.Combobox(
            tools,
            textvariable=self._sort_var,
            values=["Label", "Area", "Perimeter"],
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(tools, text="Order:").pack(side=tk.LEFT, padx=(0, 4))
        self._sort_order_var = tk.StringVar(value="Asc")
        ttk.Combobox(
            tools,
            textvariable=self._sort_order_var,
            values=["Asc", "Desc"],
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(tools, text="Label >=").pack(side=tk.LEFT, padx=(0, 4))
        self._label_min_var = tk.StringVar(value="1")
        ttk.Entry(tools, textvariable=self._label_min_var, width=6).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(tools, text="Apply Table View", command=self._apply_table_view).pack(side=tk.LEFT)

        columns = ("Label", "Area", "Perimeter", "BBox", "Centroid")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for name, width in (
            ("Label", 60),
            ("Area", 90),
            ("Perimeter", 100),
            ("BBox", 170),
            ("Centroid", 130),
        ):
            tree.heading(name, text=name)
            tree.column(name, width=width, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)

        self._tree = tree

    def _on_area_changed(self, *_args) -> None:
        self._min_area_lbl.config(text=f"{int(self._min_area_var.get())}")

    def _on_morph_open_changed(self, *_args) -> None:
        value = _ensure_odd(int(round(self._morph_open_var.get())))
        if self._morph_open_var.get() != value:
            self._morph_open_var.set(value)
        self._morph_open_lbl.config(text=str(value))

    def _on_morph_close_changed(self, *_args) -> None:
        value = _ensure_odd(int(round(self._morph_close_var.get())))
        if self._morph_close_var.get() != value:
            self._morph_close_var.set(value)
        self._morph_close_lbl.config(text=str(value))

    def on_image_loaded(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        gray = _to_gray(self.app.original_image)
        source = self._mask_var.get()
        min_area = float(self._min_area_var.get())

        if source == "Otsu":
            _, binary = otsu_threshold(gray, blur_ksize=0)
        elif source == "Adaptive Gaussian":
            binary = adaptive_threshold(gray, method="gaussian", block_size=11, c_value=2.0)
        else:
            binary = adaptive_threshold(gray, method="mean", block_size=11, c_value=2.0)

        binary = _apply_binary_morphology(
            binary,
            open_ksize=int(self._morph_open_var.get()),
            close_ksize=int(self._morph_close_var.get()),
        )

        _, _, measurements = connected_components_analysis(binary)

        filtered = [m for m in measurements if float(m["Area"]) >= min_area]
        self._all_measurements = filtered
        table_rows = self._sorted_filtered_rows(filtered)
        annotated = _draw_measurements_on_gray(gray, table_rows)

        self._summary_label.config(text=f"Detected objects: {len(table_rows)}")
        self._update_table(table_rows)

        self._show_image(gray, 0)
        self._show_image(binary, 1)
        self._show_image(annotated, 2)

        self._last_binary = binary
        self._last_annotated = annotated

    def _apply_table_view(self) -> None:
        if not self._all_measurements or self.app.original_image is None:
            return

        gray = _to_gray(self.app.original_image)
        table_rows = self._sorted_filtered_rows(self._all_measurements)
        annotated = _draw_measurements_on_gray(gray, table_rows)

        self._summary_label.config(text=f"Detected objects: {len(table_rows)}")
        self._update_table(table_rows)
        self._show_image(annotated, 2)
        self._last_annotated = annotated

    def _sorted_filtered_rows(self, measurements: list[dict]) -> list[dict]:
        label_min = _safe_int(self._label_min_var.get(), default=1)
        rows = [m for m in measurements if int(m["Label"]) >= label_min]

        reverse = self._sort_order_var.get() == "Desc"
        key_map: dict[str, str] = {
            "Label": "Label",
            "Area": "Area",
            "Perimeter": "Perimeter",
        }
        sort_key = key_map.get(self._sort_var.get(), "Label")
        rows.sort(key=lambda x: float(x[sort_key]), reverse=reverse)
        return rows

    def _on_save_annotated(self) -> None:
        if self._last_annotated is None:
            messagebox.showinfo("Save Annotated", "Run object detection first.")
            return
        _save_image_dialog(self._last_annotated, "a1_task2_detected.png")

    def _update_table(self, measurements: list[dict]) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

        for m in measurements:
            bbox = m["BBox"]
            centroid = m["Centroid"]
            self._tree.insert(
                "",
                tk.END,
                values=(
                    m["Label"],
                    int(m["Area"]),
                    f"{m['Perimeter']:.2f}",
                    f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
                    f"({centroid[0]:.1f}, {centroid[1]:.1f})",
                ),
            )

    def _show_image(self, image: np.ndarray, idx: int) -> None:
        _show_image_on_canvas(self._canvases[idx], image, self._photos, idx)


class Assignment2ObjectDetectionTab(ttk.Frame):
    """Assignment 2 tab: contour-based object detection."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app
        self._photos: list[ImageTk.PhotoImage | None] = [None, None, None]
        self._all_measurements: list[dict] = []
        self._last_binary: np.ndarray | None = None
        self._last_annotated: np.ndarray | None = None
        self._selected_labels: set[int] = set()
        self._roi_bounds: tuple[int, int, int, int] | None = None
        self._annotated_display_map: dict[str, float] | None = None
        self._drag_start_canvas: tuple[int, int] | None = None
        self._drag_rect_id: int | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        controlsButtonGroup = ttk.Frame(controls)
        controlsButtonGroup.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        self._invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Invert Binary", variable=self._invert_var).pack(
            side=tk.LEFT, padx=(0, 10)
        )

        ttk.Label(controls, text="Blur:").pack(side=tk.LEFT, padx=(0, 4))
        self._blur_var = tk.IntVar(value=5)
        for val in (3, 5, 7):
            ttk.Radiobutton(controls, text=str(val), variable=self._blur_var, value=val).pack(
                side=tk.LEFT, padx=(0, 4)
            )

        ttk.Label(controls, text="Min Area:").pack(side=tk.LEFT, padx=(10, 4))
        self._area_var = tk.DoubleVar(value=80.0)
        self._area_lbl = ttk.Label(controls, text="80", width=5)
        self._area_lbl.pack(side=tk.LEFT)
        area_scale = ttk.Scale(
            controls,
            from_=1,
            to=4000,
            orient=tk.HORIZONTAL,
            variable=self._area_var,
            length=120,
        )
        area_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._area_var.trace_add("write", self._on_area_changed)

        ttk.Label(controls, text="Min Circularity:").pack(side=tk.LEFT, padx=(0, 4))
        self._circ_var = tk.DoubleVar(value=0.30)
        self._circ_lbl = ttk.Label(controls, text="0.30", width=5)
        self._circ_lbl.pack(side=tk.LEFT)
        circ_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self._circ_var,
            length=120,
        )
        circ_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._circ_var.trace_add("write", self._on_circularity_changed)

        ttk.Label(controls, text="Morph Open:").pack(side=tk.LEFT, padx=(0, 4))
        self._morph_open_var = tk.IntVar(value=3)
        self._morph_open_lbl = ttk.Label(controls, text="3", width=3)
        self._morph_open_lbl.pack(side=tk.LEFT)
        open_scale = ttk.Scale(
            controls,
            from_=1,
            to=11,
            orient=tk.HORIZONTAL,
            variable=self._morph_open_var,
            length=100,
        )
        open_scale.pack(side=tk.LEFT, padx=(4, 8))
        self._morph_open_var.trace_add("write", self._on_morph_open_changed)

        ttk.Label(controls, text="Morph Close:").pack(side=tk.LEFT, padx=(0, 4))
        self._morph_close_var = tk.IntVar(value=3)
        self._morph_close_lbl = ttk.Label(controls, text="3", width=3)
        self._morph_close_lbl.pack(side=tk.LEFT)
        close_scale = ttk.Scale(
            controls,
            from_=1,
            to=11,
            orient=tk.HORIZONTAL,
            variable=self._morph_close_var,
            length=100,
        )
        close_scale.pack(side=tk.LEFT, padx=(4, 10))
        self._morph_close_var.trace_add("write", self._on_morph_close_changed)

        ttk.Button(controlsButtonGroup, text="Detect", command=self._refresh).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(controlsButtonGroup, text="Save Annotated", command=self._on_save_annotated).pack(side=tk.LEFT)
        ttk.Button(controlsButtonGroup, text="Save Binary", command=self._on_save_binary).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(controlsButtonGroup, text="Clear Selection", command=self._clear_selection).pack(
            side=tk.LEFT,
            padx=(6, 0),
        )

        self._interaction_label = ttk.Label(
            controls,
            text="Tip: Click object to toggle selection, or drag on Annotated view to select area.",
            font=("Helvetica", 9, "italic"),
            foreground="#f9e2af",
        )
        self._interaction_label.pack(side=tk.LEFT, padx=(10, 0))

        self._count_lbl = ttk.Label(
            controls,
            text="Objects: -",
            font=("Helvetica", 10, "bold"),
            foreground="#a6e3a1",
        )
        self._count_lbl.pack(side=tk.RIGHT)

        image_frame = ttk.Frame(self)
        image_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))

        titles = ["Original (Gray)", "Binary", "Annotated"]
        self._canvases: list[tk.Canvas] = []
        for idx, title in enumerate(titles):
            image_frame.columnconfigure(idx, weight=1)
            image_frame.rowconfigure(1, weight=1)

            ttk.Label(image_frame, text=title, font=("Helvetica", 12, "bold")).grid(
                row=0,
                column=idx,
                sticky="ew",
                padx=4,
                pady=(0, 4),
            )
            canvas = tk.Canvas(image_frame, bg="#181825", highlightthickness=0)
            canvas.grid(row=1, column=idx, sticky="nsew", padx=4)
            self._canvases.append(canvas)

        self._annotated_canvas = self._canvases[2]
        self._annotated_canvas.bind("<ButtonPress-1>", self._on_annotated_press)
        self._annotated_canvas.bind("<B1-Motion>", self._on_annotated_drag)
        self._annotated_canvas.bind("<ButtonRelease-1>", self._on_annotated_release)

        table_frame = ttk.Frame(self)
        table_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        tools = ttk.Frame(table_frame)
        tools.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(tools, text="Sort By:").pack(side=tk.LEFT, padx=(0, 4))
        self._sort_var = tk.StringVar(value="Area")
        ttk.Combobox(
            tools,
            textvariable=self._sort_var,
            values=["Label", "Area", "Circularity"],
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(tools, text="Order:").pack(side=tk.LEFT, padx=(0, 4))
        self._sort_order_var = tk.StringVar(value="Desc")
        ttk.Combobox(
            tools,
            textvariable=self._sort_order_var,
            values=["Asc", "Desc"],
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(tools, text="Label >=").pack(side=tk.LEFT, padx=(0, 4))
        self._label_min_var = tk.StringVar(value="1")
        ttk.Entry(tools, textvariable=self._label_min_var, width=6).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(tools, text="Apply Table View", command=self._apply_table_view).pack(side=tk.LEFT)

        columns = ("Label", "Area", "Circularity", "BBox", "Centroid")
        self._tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        style = ttk.Style()
        style.configure('Treeview', background="#212121", foreground="#f9e2af", fieldbackground="#212121")
        style.map('Treeview', background=[('selected', '#a6e3a1')], foreground=[('selected', '#212121')])

        for name, width in (
            ("Label", 60),
            ("Area", 90),
            ("Circularity", 100),
            ("BBox", 170),
            ("Centroid", 130),
        ):
            self._tree.heading(name, text=name)
            self._tree.column(name, width=width, anchor=tk.CENTER, )

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.configure(yscrollcommand=scrollbar.set)

    def _on_area_changed(self, *_args) -> None:
        self._area_lbl.config(text=f"{int(self._area_var.get())}")

    def _on_circularity_changed(self, *_args) -> None:
        self._circ_lbl.config(text=f"{self._circ_var.get():.2f}")

    def _on_morph_open_changed(self, *_args) -> None:
        value = _ensure_odd(int(round(self._morph_open_var.get())))
        if self._morph_open_var.get() != value:
            self._morph_open_var.set(value)
        self._morph_open_lbl.config(text=str(value))

    def _on_morph_close_changed(self, *_args) -> None:
        value = _ensure_odd(int(round(self._morph_close_var.get())))
        if self._morph_close_var.get() != value:
            self._morph_close_var.set(value)
        self._morph_close_lbl.config(text=str(value))

    def on_image_loaded(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.app.original_image is None:
            return

        self._selected_labels.clear()
        self._roi_bounds = None

        gray = _to_gray(self.app.original_image)
        binary, _annotated, measurements = contour_object_detection(
            gray,
            blur_ksize=int(self._blur_var.get()),
            invert=bool(self._invert_var.get()),
            min_area=float(self._area_var.get()),
            min_circularity=float(self._circ_var.get()),
            morph_open_ksize=int(self._morph_open_var.get()),
            morph_close_ksize=int(self._morph_close_var.get()),
        )

        self._all_measurements = measurements
        table_rows = self._interactive_rows(measurements)

        self._count_lbl.config(text=f"Objects: {len(table_rows)}")
        self._update_table(table_rows)

        annotated_for_table = _draw_measurements_on_gray(
            gray,
            table_rows,
            show_circularity=True,
            selected_labels=self._selected_labels,
            roi_bounds=self._roi_bounds,
        )

        self._show_image(gray, 0)
        self._show_image(binary, 1)
        self._show_image(annotated_for_table, 2)

        self._last_binary = binary
        self._last_annotated = annotated_for_table

    def _apply_table_view(self) -> None:
        if not self._all_measurements or self.app.original_image is None:
            return

        gray = _to_gray(self.app.original_image)
        table_rows = self._interactive_rows(self._all_measurements)
        annotated = _draw_measurements_on_gray(
            gray,
            table_rows,
            show_circularity=True,
            selected_labels=self._selected_labels,
            roi_bounds=self._roi_bounds,
        )

        self._count_lbl.config(text=f"Objects: {len(table_rows)}")
        self._update_table(table_rows)
        self._show_image(annotated, 2)
        self._last_annotated = annotated

    def _sorted_filtered_rows(self, measurements: list[dict]) -> list[dict]:
        label_min = _safe_int(self._label_min_var.get(), default=1)
        rows = [m for m in measurements if int(m["Label"]) >= label_min]

        reverse = self._sort_order_var.get() == "Desc"
        key_map: dict[str, str] = {
            "Label": "Label",
            "Area": "Area",
            "Circularity": "Circularity",
        }
        sort_key = key_map.get(self._sort_var.get(), "Area")
        rows.sort(key=lambda x: float(x[sort_key]), reverse=reverse)
        return rows

    def _interactive_rows(self, measurements: list[dict]) -> list[dict]:
        rows = self._sorted_filtered_rows(measurements)

        if self._roi_bounds is not None:
            x1, y1, x2, y2 = self._roi_bounds
            rows = [
                m for m in rows
                if x1 <= float(m["Centroid"][0]) <= x2 and y1 <= float(m["Centroid"][1]) <= y2
            ]

        if self._selected_labels:
            rows = [m for m in rows if int(m["Label"]) in self._selected_labels]

        return rows

    def _on_annotated_press(self, event: tk.Event) -> None:
        self._drag_start_canvas = (int(event.x), int(event.y))
        if self._drag_rect_id is not None:
            self._annotated_canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _on_annotated_drag(self, event: tk.Event) -> None:
        if self._drag_start_canvas is None:
            return

        x0, y0 = self._drag_start_canvas
        x1, y1 = int(event.x), int(event.y)

        if self._drag_rect_id is None:
            self._drag_rect_id = self._annotated_canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#a6e3a1",
                width=2,
                dash=(5, 3),
            )
        else:
            self._annotated_canvas.coords(self._drag_rect_id, x0, y0, x1, y1)

    def _on_annotated_release(self, event: tk.Event) -> None:
        if self._drag_start_canvas is None or self.app.original_image is None:
            return

        start_x, start_y = self._drag_start_canvas
        end_x, end_y = int(event.x), int(event.y)

        self._drag_start_canvas = None
        if self._drag_rect_id is not None:
            self._annotated_canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)

        # Treat small movement as a click (toggle object selection).
        if dx <= 4 and dy <= 4:
            point = self._canvas_to_image_point(end_x, end_y)
            if point is None:
                return

            hit = self._find_object_at_point(point[0], point[1])
            if hit is None:
                return

            label = int(hit["Label"])
            if label in self._selected_labels:
                self._selected_labels.remove(label)
            else:
                self._selected_labels.add(label)

            self._apply_table_view()
            return

        p0 = self._canvas_to_image_point(start_x, start_y)
        p1 = self._canvas_to_image_point(end_x, end_y)
        if p0 is None or p1 is None:
            return

        x1, y1 = p0
        x2, y2 = p1
        self._roi_bounds = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._selected_labels.clear()
        self._apply_table_view()

    def _canvas_to_image_point(self, cx: int, cy: int) -> tuple[int, int] | None:
        if self._annotated_display_map is None:
            return None

        m = self._annotated_display_map
        x0 = int(m["x0"])
        y0 = int(m["y0"])
        w = int(m["w"])
        h = int(m["h"])
        orig_w = int(m["orig_w"])
        orig_h = int(m["orig_h"])

        if cx < x0 or cy < y0 or cx >= x0 + w or cy >= y0 + h:
            return None

        rx = (cx - x0) / max(w, 1)
        ry = (cy - y0) / max(h, 1)
        ix = int(np.clip(rx * orig_w, 0, orig_w - 1))
        iy = int(np.clip(ry * orig_h, 0, orig_h - 1))
        return ix, iy

    def _find_object_at_point(self, px: int, py: int) -> dict | None:
        matches = []
        for m in self._all_measurements:
            x, y, w, h = m["BBox"]
            if x <= px <= x + w and y <= py <= y + h:
                matches.append(m)

        if not matches:
            return None

        matches.sort(key=lambda item: float(item["Area"]))
        return matches[0]

    def _on_save_annotated(self) -> None:
        if self._last_annotated is None:
            messagebox.showinfo("Save Annotated", "Run detection first.")
            return
        _save_image_dialog(self._last_annotated, "assignment2_detected.png")

    def _on_save_binary(self) -> None:
        if self._last_binary is None:
            messagebox.showinfo("Save Binary", "Run detection first.")
            return
        _save_image_dialog(self._last_binary, "assignment2_binary.png")

    def _clear_selection(self) -> None:
        self._selected_labels.clear()
        self._roi_bounds = None
        self._apply_table_view()

    def _update_table(self, measurements: list[dict]) -> None:
        for item in self._tree.get_children():
            self._tree.delete(item)

        for m in measurements:
            bbox = m["BBox"]
            centroid = m["Centroid"]
            self._tree.insert(
                "",
                tk.END,
                values=(
                    m["Label"],
                    f"{m['Area']:.1f}",
                    f"{m['Circularity']:.3f}",
                    f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})",
                    f"({centroid[0]:.1f}, {centroid[1]:.1f})",
                ),
            )

    def _show_image(self, image: np.ndarray, idx: int) -> None:
        canvas = self._canvases[idx]
        canvas.update_idletasks()
        canvas_w = max(canvas.winfo_width(), 180)
        canvas_h = max(canvas.winfo_height(), 180)

        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        x0 = (canvas_w - new_w) // 2
        y0 = (canvas_h - new_h) // 2

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        photo = ImageTk.PhotoImage(Image.fromarray(resized))

        canvas.delete("all")
        canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=photo)
        self._photos[idx] = photo

        if idx == 2:
            self._annotated_display_map = {
                "x0": float(x0),
                "y0": float(y0),
                "w": float(new_w),
                "h": float(new_h),
                "orig_w": float(w),
                "orig_h": float(h),
            }


class Workshop5Tab(ttk.Frame):
    """Workshop 5 parent tab that groups all workshop tasks."""

    def __init__(self, parent: tk.Widget, app) -> None:
        super().__init__(parent)
        self.app = app

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self, style="Inner.TNotebook")
        notebook.grid(row=0, column=0, sticky="nsew")

        self.thresholding_tab = ThresholdingMethodsTab(notebook, app)
        self.assignment1_tab = Assignment1Task2Tab(notebook, app)
        self.assignment2_tab = Assignment2ObjectDetectionTab(notebook, app)

        notebook.add(self.thresholding_tab, text="  Thresholding  ")
        notebook.add(self.assignment1_tab, text="  Assignment 1 - Task 2  ")
        notebook.add(self.assignment2_tab, text="  Assignment 2 - Object Detection  ")

        self._notebook = notebook
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event: tk.Event) -> None:
        if self.app.original_image is None:
            return

        selected = self._notebook.select()
        if not selected:
            return

        widget = self.nametowidget(selected)
        if hasattr(widget, "on_image_loaded"):
            widget.on_image_loaded()

    def on_image_loaded(self) -> None:
        self.thresholding_tab.on_image_loaded()
        self.assignment1_tab.on_image_loaded()
        self.assignment2_tab.on_image_loaded()


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _show_image_on_canvas(
    canvas: tk.Canvas,
    image: np.ndarray,
    photo_store: list[ImageTk.PhotoImage | None],
    idx: int,
) -> None:
    canvas.update_idletasks()
    canvas_w = max(canvas.winfo_width(), 180)
    canvas_h = max(canvas.winfo_height(), 180)

    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    scale = min(canvas_w / w, canvas_h / h, 1.0)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    photo = ImageTk.PhotoImage(Image.fromarray(resized))

    canvas.delete("all")
    canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=photo)
    photo_store[idx] = photo


def _draw_measurements_on_gray(
    gray: np.ndarray,
    measurements: list[dict],
    show_circularity: bool = False,
    selected_labels: set[int] | None = None,
    roi_bounds: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    selected = selected_labels or set()

    if roi_bounds is not None:
        x1, y1, x2, y2 = roi_bounds
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for m in measurements:
        x, y, w, h = m["BBox"]
        label = m["Label"]

        is_selected = int(label) in selected
        box_color = (0, 255, 0) if is_selected else (255, 255, 255)
        text_color = (0, 255, 255) if is_selected else (0, 0, 255)

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        text = f"{label}"
        if show_circularity and "Circularity" in m:
            text = f"{label} ({m['Circularity']:.2f})"

        cv2.putText(
            annotated,
            text,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def _apply_binary_morphology(binary: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    result = np.where(binary > 0, 255, 0).astype(np.uint8)

    open_ksize = _ensure_odd(max(int(open_ksize), 1))
    close_ksize = _ensure_odd(max(int(close_ksize), 1))

    if open_ksize > 1:
        open_kernel = np.ones((open_ksize, open_ksize), dtype=np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, open_kernel)

    if close_ksize > 1:
        close_kernel = np.ones((close_ksize, close_ksize), dtype=np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, close_kernel)

    return result


def _save_image_dialog(image: np.ndarray, default_name: str) -> None:
    initial_dir = str(_DEFAULT_OUTPUT_DIR) if _DEFAULT_OUTPUT_DIR.is_dir() else "."
    path = filedialog.asksaveasfilename(
        title="Save Result Image",
        initialdir=initial_dir,
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp")],
    )
    if not path:
        return

    ok = cv2.imwrite(path, image)
    if ok:
        messagebox.showinfo("Saved", f"Result saved to:\n{path}")
    else:
        messagebox.showerror("Save Error", f"Could not save image to:\n{path}")


def _safe_int(value: str, default: int = 1) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else default
    except ValueError:
        return default
