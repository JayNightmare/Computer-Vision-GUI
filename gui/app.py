"""
Main Application Window
=======================
Tkinter root window with a two-level tabbed notebook.
Top-level tabs: Workshop 3 · Workshop 4 · Workshop 5 · CNN.
Each contains its own sub-notebook of feature tabs.
"""

from __future__ import annotations

import io
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.figure import Figure

from .arithmetic_tab import ArithmeticTab
from .dashboard_tab import DashboardTab
from .edge_detection_tab import EdgeDetectionTab
from .filtering_tab import FilteringTab
from .histogram_tab import HistogramTab
from .cnn_tab import CNNTab
from processing import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_average_filter,
    apply_clahe,
    apply_median_filter,
    compute_histogram,
    compute_mse,
    edge_laplacian,
    edge_prewitt,
    edge_roberts,
    edge_scharr,
    edge_sobel,
    image_add,
    image_multiply,
    image_subtract,
    negative_transform,
)
from .workshop5_tab import (
    ThresholdingMethodsTab,
    Assignment1Task2Tab,
    Assignment2ObjectDetectionTab,
)

_DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "image"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


class ImageProcessorApp(tk.Tk):
    """Top-level Tk window for the image processing GUI."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Image Processing Workshop Dashboard")
        self.geometry("1400x920")
        self.minsize(1100, 750)
        self.configure(bg="#1e1e2e")

        self._style = ttk.Style(self)
        self._configure_theme()

        self.original_image = None
        self._build_menu()
        self._build_ui()

        self.after(100, self._prompt_load_image)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _configure_theme(self) -> None:
        self._style.theme_use("clam")

        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"

        self._style.configure(".", background=bg, foreground=fg, font=("Helvetica", 11))

        # --- Outer notebook (Workshop tabs) ---
        self._style.configure("TNotebook", background=bg, borderwidth=0)
        self._style.configure("TNotebook.Tab", background=surface, foreground=fg,
                              padding=[20, 10], font=("Helvetica", 12, "bold"))
        self._style.map("TNotebook.Tab",
                        background=[("selected", accent)],
                        foreground=[("selected", "#1e1e2e")])

        # --- Inner notebook (feature tabs) ---
        self._style.configure("Inner.TNotebook", background=bg, borderwidth=0)
        self._style.configure("Inner.TNotebook.Tab", background="#45475a", foreground=fg,
                              padding=[14, 7], font=("Helvetica", 11, "bold"))
        self._style.map("Inner.TNotebook.Tab",
                        background=[("selected", "#a6adc8")],
                        foreground=[("selected", "#1e1e2e")])

        self._style.configure("TFrame", background=bg)
        self._style.configure("TLabel", background=bg, foreground=fg)
        self._style.configure("TButton", background=accent, foreground="#1e1e2e",
                              font=("Helvetica", 11, "bold"), padding=[12, 6])
        self._style.map("TButton", background=[("active", "#b4d0fb")])

        self._style.configure("TScale", background=bg, troughcolor=surface)
        self._style.configure("TLabelframe", background=bg, foreground=fg)
        self._style.configure("TLabelframe.Label", background=bg, foreground=accent,
                              font=("Helvetica", 12, "bold"))
        self._style.configure("TRadiobutton", background=bg, foreground=fg,
                              font=("Helvetica", 10))
        self._style.map("TRadiobutton",
                        background=[("active", bg)],
                        foreground=[("active", accent)])

        # --- Dropdown readability (Combobox + popup list) ---
        self._style.configure(
            "TCombobox",
            fieldbackground=surface,
            background="#45475a",
            foreground=fg,
            arrowcolor=fg,
        )
        self._style.map(
            "TCombobox",
            fieldbackground=[("readonly", surface), ("!disabled", surface)],
            foreground=[("readonly", fg), ("!disabled", fg)],
            selectbackground=[("readonly", accent)],
            selectforeground=[("readonly", "#1e1e2e")],
        )

        # Popup listbox for ttk.Combobox is a Tk listbox and must be themed via option database.
        self.option_add("*TCombobox*Listbox.background", surface)
        self.option_add("*TCombobox*Listbox.foreground", fg)
        self.option_add("*TCombobox*Listbox.selectBackground", accent)
        self.option_add("*TCombobox*Listbox.selectForeground", "#1e1e2e")

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self, bg="#313244", fg="#cdd6f4",
                          activebackground="#89b4fa", activeforeground="#1e1e2e")
        file_menu = tk.Menu(menubar, tearoff=0, bg="#313244", fg="#cdd6f4",
                            activebackground="#89b4fa", activeforeground="#1e1e2e")
        file_menu.add_command(label="Open Image…", command=self._prompt_load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_quit)
        menubar.add_cascade(label="File", menu=file_menu)

        export_menu = tk.Menu(menubar, tearoff=0, bg="#313244", fg="#cdd6f4",
                              activebackground="#89b4fa", activeforeground="#1e1e2e")
        export_menu.add_command(label="Presentation Export…", command=self._on_export_presentation)
        export_menu.add_command(label="PowerPoint Export…", command=self._on_export_powerpoint)
        export_menu.add_separator()
        export_menu.add_command(label="Image Pack Export (Coming Soon)", state="disabled")
        export_menu.add_command(label="Data/Table Export (Coming Soon)", state="disabled")
        menubar.add_cascade(label="Export", menu=export_menu)

        self.config(menu=menubar)

    # ------------------------------------------------------------------
    # UI Layout — Nested Notebooks
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._inner_notebooks: list[ttk.Notebook] = []

        # ── Workshop 3 ──────────────────────────────────────────────
        ws3_frame = ttk.Frame(self.notebook)
        ws3_nb = ttk.Notebook(ws3_frame, style="Inner.TNotebook")
        ws3_nb.pack(fill=tk.BOTH, expand=True)

        self.dashboard_tab = DashboardTab(ws3_nb, self)
        self.histogram_tab = HistogramTab(ws3_nb, self)

        ws3_nb.add(self.dashboard_tab, text="  Dashboard  ")
        ws3_nb.add(self.histogram_tab, text="  Histogram  ")

        self.notebook.add(ws3_frame, text="  Workshop 3  ")
        self._ws3_nb = ws3_nb
        self._inner_notebooks.append(ws3_nb)

        # ── Workshop 4 ──────────────────────────────────────────────
        ws4_frame = ttk.Frame(self.notebook)
        ws4_nb = ttk.Notebook(ws4_frame, style="Inner.TNotebook")
        ws4_nb.pack(fill=tk.BOTH, expand=True)

        self.arithmetic_tab = ArithmeticTab(ws4_nb, self)
        self.edge_detection_tab = EdgeDetectionTab(ws4_nb, self)
        self.filtering_tab = FilteringTab(ws4_nb, self)

        ws4_nb.add(self.arithmetic_tab, text="  Arithmetic  ")
        ws4_nb.add(self.edge_detection_tab, text="  Edge Detection  ")
        ws4_nb.add(self.filtering_tab, text="  Filtering  ")

        self.notebook.add(ws4_frame, text="  Workshop 4  ")
        self._ws4_nb = ws4_nb
        self._inner_notebooks.append(ws4_nb)

        # ── Workshop 5 ──────────────────────────────────────────────
        ws5_frame = ttk.Frame(self.notebook)
        ws5_nb = ttk.Notebook(ws5_frame, style="Inner.TNotebook")
        ws5_nb.pack(fill=tk.BOTH, expand=True)

        self.ws5_thresholding_tab = ThresholdingMethodsTab(ws5_nb, self)
        self.ws5_assignment1_tab = Assignment1Task2Tab(ws5_nb, self)
        self.ws5_assignment2_tab = Assignment2ObjectDetectionTab(ws5_nb, self)

        ws5_nb.add(self.ws5_thresholding_tab, text="  Thresholding  ")
        ws5_nb.add(self.ws5_assignment1_tab, text="  Assignment 1 - Task 2  ")
        ws5_nb.add(self.ws5_assignment2_tab, text="  Assignment 2 - Object Detection  ")

        self.notebook.add(ws5_frame, text="  Workshop 5  ")
        self._ws5_nb = ws5_nb
        self._inner_notebooks.append(ws5_nb)

        # ── CNN ─────────────────────────────────────────────────────
        cnn_frame = ttk.Frame(self.notebook)
        cnn_nb = ttk.Notebook(cnn_frame, style="Inner.TNotebook")
        cnn_nb.pack(fill=tk.BOTH, expand=True)

        self.cnn_tab = CNNTab(cnn_nb, self)
        cnn_nb.add(self.cnn_tab, text="  Workshop 1 - Basic CNN  ")

        self.notebook.add(cnn_frame, text="  CNN  ")
        self._cnn_nb = cnn_nb
        self._inner_notebooks.append(cnn_nb)

        # ── Tab-change events ────────────────────────────────────────
        self._ws3_nb.bind("<<NotebookTabChanged>>", self._on_inner_tab_changed)
        self._ws4_nb.bind("<<NotebookTabChanged>>", self._on_inner_tab_changed)
        self._ws5_nb.bind("<<NotebookTabChanged>>", self._on_inner_tab_changed)
        self._cnn_nb.bind("<<NotebookTabChanged>>", self._on_inner_tab_changed)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_outer_tab_changed)

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))

        open_btn = ttk.Button(toolbar, text="📂  Open Image",
                              command=self._prompt_load_image)
        open_btn.pack(side=tk.LEFT)

        export_current_btn = ttk.Button(
            toolbar,
            text="📝  Export Current",
            command=self._on_export_current_tab,
        )
        export_current_btn.pack(side=tk.LEFT, padx=(8, 0))

        self._filename_label = ttk.Label(
            toolbar, text="No image loaded",
            font=("Helvetica", 11, "italic"), foreground="#6c7086",
        )
        self._filename_label.pack(side=tk.LEFT, padx=(16, 0))

    # ------------------------------------------------------------------
    # Tab-change Handlers
    # ------------------------------------------------------------------

    def _on_outer_tab_changed(self, _event: tk.Event) -> None:
        if self.original_image is None:
            return
        self._refresh_active_inner_tab()

    def _on_inner_tab_changed(self, _event: tk.Event) -> None:
        if self.original_image is None:
            return
        self._refresh_active_inner_tab()

    def _refresh_active_inner_tab(self) -> None:
        outer_sel = self.notebook.select()
        if not outer_sel:
            return

        outer_idx = self.notebook.index(outer_sel)
        if outer_idx >= len(self._inner_notebooks):
            return

        inner_nb = self._inner_notebooks[outer_idx]

        inner_sel = inner_nb.select()
        if not inner_sel:
            return

        widget = self.nametowidget(inner_sel)
        if hasattr(widget, "on_image_loaded"):
            widget.on_image_loaded()

    # ------------------------------------------------------------------
    # Image Loading
    # ------------------------------------------------------------------

    def _prompt_load_image(self) -> None:
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("All files", "*.*"),
        ]
        initial_dir = str(_DEFAULT_IMAGE_DIR) if _DEFAULT_IMAGE_DIR.is_dir() else "."
        path = filedialog.askopenfilename(
            title="Select an image",
            initialdir=initial_dir,
            filetypes=filetypes,
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Failed to read image:\n{path}")
            return

        self.original_image = img
        filename = Path(path).name
        self.title(f"Image Processing Workshop — {filename}")
        self._filename_label.config(text=f"📷  {filename}", foreground="#a6e3a1")

        for tab in (self.dashboard_tab, self.histogram_tab,
                    self.arithmetic_tab, self.edge_detection_tab,
                self.filtering_tab, self.ws5_thresholding_tab,
                self.ws5_assignment1_tab, self.ws5_assignment2_tab,
                self.cnn_tab):
            tab.on_image_loaded()

    def _on_quit(self) -> None:
        self.destroy()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_presentation(self) -> None:
        selection = self._prompt_export_selection()
        if selection is None:
            return

        workshop_name, subtabs = selection
        if workshop_name != "CNN" and self.original_image is None:
            messagebox.showwarning("No image", "Load an image before exporting this workshop.")
            return

        if not subtabs:
            messagebox.showwarning("Export", "No subtab selected for export.")
            return

        self._run_presentation_export(workshop_name, subtabs)

    def _on_export_powerpoint(self) -> None:
        selection = self._prompt_export_selection()
        if selection is None:
            return

        workshop_name, subtabs = selection
        if workshop_name != "CNN" and self.original_image is None:
            messagebox.showwarning("No image", "Load an image before exporting this workshop.")
            return

        if not subtabs:
            messagebox.showwarning("Export", "No subtab selected for export.")
            return

        self._run_powerpoint_export(workshop_name, subtabs)

    def _on_export_current_tab(self) -> None:
        selection = self._get_current_tab_selection()
        if selection is None:
            messagebox.showwarning("Export", "No active tab found to export.")
            return

        workshop_name, subtab_name = selection
        if workshop_name != "CNN" and self.original_image is None:
            messagebox.showwarning("No image", "Load an image before exporting the current tab.")
            return

        self._run_presentation_export(workshop_name, [subtab_name])

    def _run_presentation_export(self, workshop_name: str, subtabs: list[str]) -> None:
        initial_dir = str(_DEFAULT_OUTPUT_DIR) if _DEFAULT_OUTPUT_DIR.is_dir() else "."
        default_file = f"presentation_{workshop_name.replace(' ', '_').lower()}.docx"
        save_path = filedialog.asksaveasfilename(
            title="Save presentation DOCX",
            initialdir=initial_dir,
            initialfile=default_file,
            defaultextension=".docx",
            filetypes=[("Word Document", "*.docx")],
        )
        if not save_path:
            return

        try:
            from docx import Document
            from docx.shared import Inches
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "python-docx is required for Presentation Export.\n"
                "Install it with: pip install python-docx",
            )
            return

        try:
            self._export_figure_index = 0
            document = Document()
            self._build_export_cover_and_toc(document, workshop_name, subtabs)

            for idx, subtab_name in enumerate(subtabs):
                self._append_subtab_export(document, Inches, workshop_name, subtab_name)
                if idx < len(subtabs) - 1:
                    document.add_page_break()

            document.save(save_path)
            messagebox.showinfo("Export complete", f"Presentation exported to:\n{save_path}")
        except Exception as exc:  # pragma: no cover - defensive UI error handling
            messagebox.showerror("Export failed", f"Failed to export presentation:\n{exc}")

    def _run_powerpoint_export(self, workshop_name: str, subtabs: list[str]) -> None:
        initial_dir = str(_DEFAULT_OUTPUT_DIR) if _DEFAULT_OUTPUT_DIR.is_dir() else "."
        default_file = f"presentation_{workshop_name.replace(' ', '_').lower()}.pptx"
        save_path = filedialog.asksaveasfilename(
            title="Save presentation PPTX",
            initialdir=initial_dir,
            initialfile=default_file,
            defaultextension=".pptx",
            filetypes=[("PowerPoint Presentation", "*.pptx")],
        )
        if not save_path:
            return

        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "python-pptx is required for PowerPoint Export.\n"
                "Install it with: pip install python-pptx",
            )
            return

        try:
            prs = Presentation()

            title_slide = prs.slides.add_slide(prs.slide_layouts[0])
            title_slide.shapes.title.text = workshop_name
            title_slide.placeholders[1].text = (
                "Image Processing Workshop Presentation Export\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            toc_slide = prs.slides.add_slide(prs.slide_layouts[1])
            toc_slide.shapes.title.text = "Table of Contents"
            tf = toc_slide.placeholders[1].text_frame
            tf.clear()
            for subtab in subtabs:
                p = tf.add_paragraph()
                p.text = subtab
                p.level = 0

            for subtab_name in subtabs:
                self._append_subtab_export_ppt(prs, workshop_name, subtab_name, Inches)

            prs.save(save_path)
            messagebox.showinfo("Export complete", f"PowerPoint exported to:\n{save_path}")
        except Exception as exc:  # pragma: no cover - defensive UI error handling
            messagebox.showerror("Export failed", f"Failed to export PowerPoint:\n{exc}")

    def _prompt_export_selection(self) -> tuple[str, list[str]] | None:
        workshops = {
            "Workshop 3": ["Dashboard", "Histogram"],
            "Workshop 4": ["Arithmetic", "Edge Detection", "Filtering"],
            "Workshop 5": [
                "Thresholding",
                "Assignment 1 - Task 2",
                "Assignment 2 - Object Detection",
            ],
            "CNN": ["Workshop 1 - Basic CNN"],
        }

        result: dict[str, object] = {"ok": False}
        dialog = tk.Toplevel(self)
        dialog.title("Presentation Export")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        workshop_var = tk.StringVar(value="Workshop 3")
        include_all_var = tk.BooleanVar(value=True)

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frame, text="Workshop:", font=("Helvetica", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        workshop_combo = ttk.Combobox(
            frame,
            textvariable=workshop_var,
            values=list(workshops.keys()),
            state="readonly",
            width=30,
        )
        workshop_combo.grid(row=1, column=0, sticky="ew", pady=(4, 8))

        include_all_chk = ttk.Checkbutton(
            frame,
            text="Include all subtabs in selected workshop",
            variable=include_all_var,
        )
        include_all_chk.grid(row=2, column=0, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Subtab:", font=("Helvetica", 11, "bold")).grid(
            row=3, column=0, sticky="w"
        )
        subtab_combo = ttk.Combobox(frame, state="readonly", width=30)
        subtab_combo.grid(row=4, column=0, sticky="ew", pady=(4, 12))

        def _update_subtabs(*_args) -> None:
            values = workshops[workshop_var.get()]
            subtab_combo.configure(values=values)
            if values:
                subtab_combo.set(values[0])
            state = "disabled" if include_all_var.get() else "readonly"
            subtab_combo.configure(state=state)

        workshop_var.trace_add("write", _update_subtabs)
        include_all_var.trace_add("write", _update_subtabs)
        _update_subtabs()

        btns = ttk.Frame(frame)
        btns.grid(row=5, column=0, sticky="e")

        def _confirm() -> None:
            selected_workshop = workshop_var.get()
            if include_all_var.get():
                selected_subtabs = list(workshops[selected_workshop])
            else:
                selected_subtab = subtab_combo.get().strip()
                selected_subtabs = [selected_subtab] if selected_subtab else []

            result["ok"] = True
            result["workshop"] = selected_workshop
            result["subtabs"] = selected_subtabs
            dialog.destroy()

        def _cancel() -> None:
            dialog.destroy()

        ttk.Button(btns, text="Cancel", command=_cancel).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns, text="Export", command=_confirm).pack(side=tk.RIGHT)

        dialog.wait_window()
        if not result.get("ok"):
            return None

        return str(result["workshop"]), list(result["subtabs"])

    def _get_current_tab_selection(self) -> tuple[str, str] | None:
        outer_sel = self.notebook.select()
        if not outer_sel:
            return None

        outer_idx = self.notebook.index(outer_sel)
        if outer_idx >= len(self._inner_notebooks):
            return None

        workshop_name = self.notebook.tab(outer_sel, "text").strip()
        inner_nb = self._inner_notebooks[outer_idx]
        inner_sel = inner_nb.select()
        if not inner_sel:
            return None

        subtab_name = inner_nb.tab(inner_sel, "text").strip()
        return workshop_name, subtab_name

    def _build_export_cover_and_toc(self, document, workshop_name: str, subtabs: list[str]) -> None:
        document.add_heading(workshop_name, level=0)
        document.add_paragraph("Image Processing Workshop Presentation Export")
        document.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        document.add_paragraph("Included sections: " + ", ".join(subtabs))

        document.add_page_break()
        document.add_heading("Table of Contents", level=1)
        self._insert_toc_field(document)
        document.add_paragraph("Open this file in Word and update fields to render TOC page numbers.")
        document.add_page_break()

    @staticmethod
    def _insert_toc_field(document) -> None:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        paragraph = document.add_paragraph()
        run = paragraph.add_run()

        fld_begin = OxmlElement("w:fldChar")
        fld_begin.set(qn("w:fldCharType"), "begin")

        instr_text = OxmlElement("w:instrText")
        instr_text.set(qn("xml:space"), "preserve")
        instr_text.text = 'TOC \\o "1-3" \\h \\z \\u'

        fld_separate = OxmlElement("w:fldChar")
        fld_separate.set(qn("w:fldCharType"), "separate")

        fld_text = OxmlElement("w:t")
        fld_text.text = "Table of contents will appear here after updating fields."

        fld_end = OxmlElement("w:fldChar")
        fld_end.set(qn("w:fldCharType"), "end")

        run._r.append(fld_begin)
        run._r.append(instr_text)
        run._r.append(fld_separate)
        run._r.append(fld_text)
        run._r.append(fld_end)

    def _append_subtab_export(self, document, Inches, workshop_name: str, subtab_name: str) -> None:
        document.add_heading(subtab_name, level=1)

        if workshop_name == "Workshop 3" and subtab_name == "Dashboard":
            self._export_ws3_dashboard(document, Inches)
            return
        if workshop_name == "Workshop 3" and subtab_name == "Histogram":
            self._export_ws3_histogram(document, Inches)
            return
        if workshop_name == "Workshop 4" and subtab_name == "Arithmetic":
            self._export_ws4_arithmetic(document, Inches)
            return
        if workshop_name == "Workshop 4" and subtab_name == "Edge Detection":
            self._export_ws4_edge_detection(document, Inches)
            return
        if workshop_name == "Workshop 4" and subtab_name == "Filtering":
            self._export_ws4_filtering(document, Inches)
            return
        if workshop_name == "Workshop 5" and subtab_name == "Thresholding":
            self._export_ws5_thresholding(document, Inches)
            return
        if workshop_name == "Workshop 5" and subtab_name == "Assignment 1 - Task 2":
            self._export_ws5_assignment1_task2(document, Inches)
            return
        if workshop_name == "Workshop 5" and subtab_name == "Assignment 2 - Object Detection":
            self._export_ws5_assignment2(document, Inches)
            return
        if workshop_name == "CNN" and subtab_name == "Workshop 1 - Basic CNN":
            self._export_cnn(document, Inches)
            return

        document.add_paragraph("No exporter configured for this tab.")
        self._add_image_to_doc(document, self.original_image, Inches(5.8), "Current image")

    def _append_subtab_export_ppt(self, prs, workshop_name: str, subtab_name: str, Inches) -> None:
        if workshop_name == "Workshop 3" and subtab_name == "Dashboard":
            tab = self.dashboard_tab
            altered = tab._compute_altered()
            settings = [
                f"Brightness: {tab.brightness_var.get():.2f}",
                f"Contrast: {tab.contrast_var.get():.2f}",
                f"Gamma: {tab.gamma_var.get():.2f}",
                f"Equalize: {'On' if tab._equalized else 'Off'}",
                f"Grayscale: {'On' if tab._grayscale else 'Off'}",
            ]
            self._ppt_add_summary_slide(prs, f"{subtab_name} - Summary", settings)
            self._ppt_add_image_slide(prs, subtab_name, "Original image", self.original_image, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Altered image", altered, Inches)
            self._ppt_add_stream_slide(prs, subtab_name, "Original histogram", self._hist_figure_png(self.original_image, "Original Histogram"), Inches)
            self._ppt_add_stream_slide(prs, subtab_name, "Altered histogram", self._hist_figure_png(altered, "Altered Histogram"), Inches)
            return

        if workshop_name == "Workshop 3" and subtab_name == "Histogram":
            current = self.histogram_tab._get_current_image()
            clahe_img = apply_clahe(current)
            negative_img = negative_transform(current)
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - Summary",
                ["Views: Current, CLAHE, Negative", "Includes image and histogram for each view"],
            )
            self._ppt_add_image_slide(prs, subtab_name, "Current image", current, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "CLAHE image", clahe_img, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Negative image", negative_img, Inches)
            self._ppt_add_stream_slide(prs, subtab_name, "Current histogram", self._hist_figure_png(current, "Current Histogram"), Inches)
            self._ppt_add_stream_slide(prs, subtab_name, "CLAHE histogram", self._hist_figure_png(clahe_img, "CLAHE Histogram"), Inches)
            self._ppt_add_stream_slide(prs, subtab_name, "Negative histogram", self._hist_figure_png(negative_img, "Negative Histogram"), Inches)
            return

        if workshop_name == "Workshop 4" and subtab_name == "Arithmetic":
            tab = self.arithmetic_tab
            operation = tab._op_var.get()
            self._ppt_add_summary_slide(prs, f"{subtab_name} - Summary", [f"Operation: {operation}"])
            self._ppt_add_image_slide(prs, subtab_name, "Image A", self.original_image, Inches)
            if tab._image_b is None:
                self._ppt_add_summary_slide(prs, subtab_name, ["Image B not loaded"]) 
                return
            self._ppt_add_image_slide(prs, subtab_name, "Image B", tab._image_b, Inches)
            ops = {"Add": image_add, "Subtract": image_subtract, "Multiply": image_multiply}
            result = ops.get(operation, image_add)(self.original_image, tab._image_b)
            self._ppt_add_image_slide(prs, subtab_name, f"Result ({operation})", result, Inches)
            return

        if workshop_name == "Workshop 4" and subtab_name == "Edge Detection":
            self._ppt_add_summary_slide(prs, f"{subtab_name} - Summary", ["Sobel, Scharr, Laplacian, Roberts, Prewitt"])
            for name, fn in (
                ("Sobel", edge_sobel),
                ("Scharr", edge_scharr),
                ("Laplacian", edge_laplacian),
                ("Roberts", edge_roberts),
                ("Prewitt", edge_prewitt),
            ):
                self._ppt_add_image_slide(prs, subtab_name, name, fn(self.original_image), Inches)
            return

        if workshop_name == "Workshop 4" and subtab_name == "Filtering":
            tab = self.filtering_tab
            noise_type = tab._noise_var.get()
            strength = float(tab._noise_strength_var.get())
            filter_type = tab._filter_var.get()
            ksize = int(tab._ksize_var.get())
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - Summary",
                [
                    f"Noise: {noise_type} (strength={strength:.2f})",
                    f"Filter: {filter_type} (kernel={ksize})",
                ],
            )
            noisy = tab._noisy_image
            if noisy is None:
                noisy = add_gaussian_noise(self.original_image, std=strength) if noise_type == "Gaussian" else add_salt_pepper_noise(self.original_image, amount=strength / 500.0)
            filtered = apply_average_filter(noisy, ksize) if filter_type == "Average" else apply_median_filter(noisy, ksize)
            self._ppt_add_image_slide(prs, subtab_name, "Original", self.original_image, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Noisy", noisy, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Filtered", filtered, Inches)
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - MSE",
                [
                    f"MSE(noisy vs original): {compute_mse(self.original_image, noisy):.2f}",
                    f"MSE(filtered vs original): {compute_mse(self.original_image, filtered):.2f}",
                ],
            )
            return

        if workshop_name == "Workshop 5" and subtab_name == "Thresholding":
            tab = self.ws5_thresholding_tab
            tab._refresh()
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - Summary",
                [
                    f"Adaptive Method: {tab._adaptive_method_var.get()}",
                    f"Block Size: {int(tab._block_var.get())}, C: {float(tab._c_var.get()):.1f}",
                    f"Otsu Blur: {int(tab._blur_var.get())}",
                ],
            )
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self._ppt_add_image_slide(prs, subtab_name, "Original gray", gray, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Adaptive", tab._last_outputs.get("adaptive"), Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Otsu raw", tab._last_outputs.get("otsu_raw"), Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Otsu blur", tab._last_outputs.get("otsu_blur"), Inches)
            return

        if workshop_name == "Workshop 5" and subtab_name == "Assignment 1 - Task 2":
            seg = self.ws5_assignment1_tab.segmentation_tab
            seg._refresh()
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - Segmentation",
                [
                    f"Method: {seg._method_var.get()}",
                    f"Block: {int(seg._block_var.get())}, C: {float(seg._c_var.get()):.1f}",
                ],
            )
            self._ppt_add_image_slide(prs, subtab_name, "Segmentation - Original gray", gray, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Segmentation - Binary", seg._last_binary, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Segmentation - Labeled", seg._last_labeled, Inches)

            od = self.ws5_assignment1_tab.object_detection_tab
            od._refresh()
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - Object Detection",
                [
                    f"Mask Source: {od._mask_var.get()}",
                    f"Min Area: {float(od._min_area_var.get()):.1f}",
                    f"Morph Open: {int(od._morph_open_var.get())}, Morph Close: {int(od._morph_close_var.get())}",
                ],
            )
            self._ppt_add_image_slide(prs, subtab_name, "Detection - Binary", od._last_binary, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Detection - Annotated", od._last_annotated, Inches)
            headers, rows = self._treeview_rows(od._tree)
            self._ppt_add_table_slide(prs, f"{subtab_name} - Detection Table", headers, rows)
            return

        if workshop_name == "Workshop 5" and subtab_name == "Assignment 2 - Object Detection":
            tab = self.ws5_assignment2_tab
            tab._refresh()
            self._ppt_add_summary_slide(
                prs,
                f"{subtab_name} - Settings",
                [
                    f"Invert: {'Yes' if tab._invert_var.get() else 'No'}, Blur: {int(tab._blur_var.get())}",
                    f"Min Area: {float(tab._area_var.get()):.1f}, Min Circularity: {float(tab._circ_var.get()):.2f}",
                    f"Morph Open: {int(tab._morph_open_var.get())}, Morph Close: {int(tab._morph_close_var.get())}",
                    f"Table sort: {tab._sort_var.get()} {tab._sort_order_var.get()}, label >= {tab._label_min_var.get()}",
                ],
            )
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self._ppt_add_image_slide(prs, subtab_name, "Original gray", gray, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Binary", tab._last_binary, Inches)
            self._ppt_add_image_slide(prs, subtab_name, "Annotated", tab._last_annotated, Inches)
            headers, rows = self._treeview_rows(tab._tree)
            self._ppt_add_table_slide(prs, f"{subtab_name} - Detection Table", headers, rows)
            return

        if workshop_name == "CNN" and subtab_name == "Workshop 1 - Basic CNN":
            self._export_cnn_ppt(prs, subtab_name, Inches)
            return

        self._ppt_add_summary_slide(prs, subtab_name, ["No PowerPoint exporter configured for this tab."])
        self._ppt_add_image_slide(prs, subtab_name, "Current image", self.original_image, Inches)

    @staticmethod
    def _ppt_add_summary_slide(prs, title: str, bullets: list[str]) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        tf = slide.placeholders[1].text_frame
        tf.clear()
        for idx, line in enumerate(bullets):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = line
            p.level = 0

    @staticmethod
    def _ppt_add_image_slide(prs, title: str, caption: str, image: np.ndarray | None, Inches) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title

        if image is None:
            tx = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.5), Inches(1.0))
            tx.text_frame.text = f"{caption}: not available"
            return

        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            tx = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.5), Inches(1.0))
            tx.text_frame.text = f"{caption}: failed to encode"
            return

        stream = io.BytesIO(encoded.tobytes())
        slide.shapes.add_picture(stream, Inches(0.8), Inches(1.2), width=Inches(11.5), height=Inches(5.8))
        cap = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(11.5), Inches(0.5))
        cap.text_frame.text = caption

    @staticmethod
    def _ppt_add_stream_slide(prs, title: str, caption: str, stream: io.BytesIO, Inches) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title
        stream.seek(0)
        slide.shapes.add_picture(stream, Inches(0.8), Inches(1.2), width=Inches(11.5), height=Inches(5.8))
        cap = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(11.5), Inches(0.5))
        cap.text_frame.text = caption

    @staticmethod
    def _treeview_rows(tree) -> tuple[list[str], list[list[str]]]:
        headers = [str(col) for col in tree["columns"]]
        rows: list[list[str]] = []
        for item_id in tree.get_children():
            values = tree.item(item_id).get("values", [])
            rows.append([str(v) for v in values])
        return headers, rows

    @staticmethod
    def _ppt_add_table_slide(prs, title: str, headers: list[str], rows: list[list[str]]) -> None:
        from pptx.util import Inches

        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title

        if not headers:
            tx = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.5), Inches(1.0))
            tx.text_frame.text = "No table columns available"
            return

        max_rows = 14
        display_rows = rows[:max_rows]
        row_count = len(display_rows) + 1
        col_count = len(headers)

        table_shape = slide.shapes.add_table(
            row_count,
            col_count,
            Inches(0.5),
            Inches(1.2),
            Inches(12.3),
            Inches(5.9),
        )
        table = table_shape.table

        for col_idx, header in enumerate(headers):
            table.cell(0, col_idx).text = header

        for row_idx, row_values in enumerate(display_rows, start=1):
            for col_idx, value in enumerate(row_values):
                if col_idx < col_count:
                    table.cell(row_idx, col_idx).text = value

        if len(rows) > max_rows:
            tx = slide.shapes.add_textbox(Inches(0.8), Inches(7.2), Inches(11.5), Inches(0.4))
            tx.text_frame.text = f"Showing first {max_rows} rows out of {len(rows)}"

    def _export_ws3_dashboard(self, document, Inches) -> None:
        tab = self.dashboard_tab
        altered = tab._compute_altered()

        settings = [
            f"Brightness: {tab.brightness_var.get():.2f}",
            f"Contrast: {tab.contrast_var.get():.2f}",
            f"Gamma: {tab.gamma_var.get():.2f}",
            f"Equalize: {'On' if tab._equalized else 'Off'}",
            f"Grayscale: {'On' if tab._grayscale else 'Off'}",
        ]
        document.add_paragraph("Adjustment settings: " + ", ".join(settings))

        self._add_image_to_doc(document, self.original_image, Inches(5.8), "Dashboard - Original image")
        self._add_image_to_doc(document, altered, Inches(5.8), "Dashboard - Altered image")

        orig_hist = self._hist_figure_png(self.original_image, "Original Histogram")
        alt_hist = self._hist_figure_png(altered, "Altered Histogram")
        self._add_stream_figure(document, orig_hist, Inches(5.8), "Dashboard - Original histogram")
        self._add_stream_figure(document, alt_hist, Inches(5.8), "Dashboard - Altered histogram")

    def _export_ws3_histogram(self, document, Inches) -> None:
        current = self.histogram_tab._get_current_image()
        clahe_img = apply_clahe(current)
        negative_img = negative_transform(current)

        document.add_paragraph(
            "Views exported: Current (dashboard-adjusted), CLAHE, and Negative."
        )

        images = [
            ("Current image", current),
            ("CLAHE image", clahe_img),
            ("Negative image", negative_img),
        ]
        for caption, img in images:
            self._add_image_to_doc(document, img, Inches(5.8), f"Histogram tab - {caption}")

        hists = [
            ("Current Histogram", current),
            ("CLAHE Histogram", clahe_img),
            ("Negative Histogram", negative_img),
        ]
        for title, img in hists:
            stream = self._hist_figure_png(img, title)
            self._add_stream_figure(document, stream, Inches(5.8), f"Histogram tab - {title}")

    def _export_ws4_arithmetic(self, document, Inches) -> None:
        tab = self.arithmetic_tab
        image_a = self.original_image
        image_b = tab._image_b
        operation = tab._op_var.get()

        document.add_paragraph(f"Operation: {operation}")
        self._add_image_to_doc(document, image_a, Inches(5.8), "Arithmetic - Image A")

        if image_b is None:
            document.add_paragraph("Image B is not loaded; result cannot be generated.")
            return

        self._add_image_to_doc(document, image_b, Inches(5.8), "Arithmetic - Image B")
        ops = {"Add": image_add, "Subtract": image_subtract, "Multiply": image_multiply}
        result = ops.get(operation, image_add)(image_a, image_b)
        self._add_image_to_doc(document, result, Inches(5.8), f"Arithmetic - Result ({operation})")

    def _export_ws4_edge_detection(self, document, Inches) -> None:
        detectors = [
            ("Sobel", edge_sobel),
            ("Scharr", edge_scharr),
            ("Laplacian", edge_laplacian),
            ("Roberts", edge_roberts),
            ("Prewitt", edge_prewitt),
        ]
        document.add_paragraph("Edge operators exported on the currently loaded image.")
        for name, fn in detectors:
            edges = fn(self.original_image)
            self._add_image_to_doc(document, edges, Inches(5.8), f"Edge Detection - {name}")

    def _export_ws4_filtering(self, document, Inches) -> None:
        tab = self.filtering_tab
        original = self.original_image

        noise_type = tab._noise_var.get()
        strength = float(tab._noise_strength_var.get())
        filter_type = tab._filter_var.get()
        ksize = int(tab._ksize_var.get())

        document.add_paragraph(
            f"Noise: {noise_type} (strength={strength:.2f}), Filter: {filter_type} (kernel={ksize})"
        )
        self._add_image_to_doc(document, original, Inches(5.8), "Filtering - Original")

        noisy = tab._noisy_image
        if noisy is None:
            if noise_type == "Gaussian":
                noisy = add_gaussian_noise(original, std=strength)
            else:
                noisy = add_salt_pepper_noise(original, amount=strength / 500.0)

        self._add_image_to_doc(document, noisy, Inches(5.8), "Filtering - Noisy")

        if filter_type == "Average":
            filtered = apply_average_filter(noisy, ksize)
        else:
            filtered = apply_median_filter(noisy, ksize)

        self._add_image_to_doc(document, filtered, Inches(5.8), "Filtering - Filtered")

        mse_noisy = compute_mse(original, noisy)
        mse_filtered = compute_mse(original, filtered)
        document.add_paragraph(
            f"MSE(noisy vs original): {mse_noisy:.2f} | "
            f"MSE(filtered vs original): {mse_filtered:.2f}"
        )

    def _export_ws5_thresholding(self, document, Inches) -> None:
        tab = self.ws5_thresholding_tab
        tab._refresh()

        method = tab._adaptive_method_var.get()
        block = int(tab._block_var.get())
        c_value = float(tab._c_var.get())
        blur = int(tab._blur_var.get())

        document.add_paragraph(
            f"Adaptive Method: {method}, Block Size: {block}, C: {c_value:.1f}, Otsu Blur: {blur}"
        )

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self._add_image_to_doc(document, gray, Inches(5.8), "Thresholding - Original gray")
        self._add_image_to_doc(document, tab._last_outputs.get("adaptive"), Inches(5.8), "Thresholding - Adaptive")
        self._add_image_to_doc(document, tab._last_outputs.get("otsu_raw"), Inches(5.8), "Thresholding - Otsu raw")
        self._add_image_to_doc(document, tab._last_outputs.get("otsu_blur"), Inches(5.8), "Thresholding - Otsu blur")

    def _export_ws5_assignment1_task2(self, document, Inches) -> None:
        parent = self.ws5_assignment1_tab

        seg = parent.segmentation_tab
        seg._refresh()
        document.add_heading("Segmentation", level=2)
        document.add_paragraph(
            f"Method: {seg._method_var.get()}, Block: {int(seg._block_var.get())}, C: {float(seg._c_var.get()):.1f}"
        )
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self._add_image_to_doc(document, gray, Inches(5.8), "A1 Task2 Segmentation - Original gray")
        self._add_image_to_doc(document, seg._last_binary, Inches(5.8), "A1 Task2 Segmentation - Binary mask")
        self._add_image_to_doc(document, seg._last_labeled, Inches(5.8), "A1 Task2 Segmentation - Labeled regions")

        od = parent.object_detection_tab
        od._refresh()
        document.add_heading("Object Detection", level=2)
        document.add_paragraph(
            f"Mask Source: {od._mask_var.get()}, Min Area: {float(od._min_area_var.get()):.1f}, "
            f"Morph Open: {int(od._morph_open_var.get())}, Morph Close: {int(od._morph_close_var.get())}"
        )
        self._add_image_to_doc(document, gray, Inches(5.8), "A1 Task2 Detection - Original gray")
        self._add_image_to_doc(document, od._last_binary, Inches(5.8), "A1 Task2 Detection - Binary mask")
        self._add_image_to_doc(document, od._last_annotated, Inches(5.8), "A1 Task2 Detection - Annotated")
        self._add_treeview_table_to_doc(document, od._tree, "A1 Task2 Detection table")

    def _export_ws5_assignment2(self, document, Inches) -> None:
        tab = self.ws5_assignment2_tab
        tab._refresh()

        document.add_paragraph(
            f"Invert: {'Yes' if tab._invert_var.get() else 'No'}, Blur: {int(tab._blur_var.get())}, "
            f"Min Area: {float(tab._area_var.get()):.1f}, Min Circularity: {float(tab._circ_var.get()):.2f}, "
            f"Morph Open: {int(tab._morph_open_var.get())}, Morph Close: {int(tab._morph_close_var.get())}"
        )
        document.add_paragraph(
            f"Table View: sort={tab._sort_var.get()} {tab._sort_order_var.get()}, "
            f"label >= {tab._label_min_var.get()}"
        )

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self._add_image_to_doc(document, gray, Inches(5.8), "A2 Detection - Original gray")
        self._add_image_to_doc(document, tab._last_binary, Inches(5.8), "A2 Detection - Binary")
        self._add_image_to_doc(document, tab._last_annotated, Inches(5.8), "A2 Detection - Annotated")
        self._add_treeview_table_to_doc(document, tab._tree, "A2 Detection table")

    def _export_cnn(self, document, Inches) -> None:
        tab = self.cnn_tab

        document.add_paragraph("Workshop 1 - CNN classifier for synthetic circles and squares")
        document.add_paragraph(
            "Dataset settings: "
            f"image_size={int(tab.image_size_var.get())}, "
            f"samples/class={int(tab.samples_per_class_var.get())}, "
            f"test_split={float(tab.test_split_var.get()):.2f}, "
            f"rotation={'on' if tab.apply_rotation_var.get() else 'off'} ({float(tab.rotation_range_var.get()):.1f}), "
            f"noise={'on' if tab.apply_noise_var.get() else 'off'} (std={float(tab.noise_std_var.get()):.1f}), "
            f"seed={int(tab.seed_var.get())}"
        )
        document.add_paragraph(
            "Training settings: "
            f"mode={tab.mode_var.get()}, epochs={int(tab.epochs_var.get())}, "
            f"batch_size={int(tab.batch_size_var.get())}, tf_logs={tab.tf_log_level_var.get()}"
        )

        if tab.metrics:
            document.add_paragraph(
                f"Evaluation: accuracy={tab.metrics['accuracy'] * 100:.2f}% | loss={tab.metrics['loss']:.4f}"
            )
        else:
            document.add_paragraph("Evaluation: not available yet (run Train + Evaluate in CNN tab).")

        history_stream = self._cnn_history_figure_png()
        if history_stream is not None:
            self._add_stream_figure(document, history_stream, Inches(5.8), "CNN - Training curves")
        else:
            document.add_paragraph("CNN - Training curves: not available")

        montage = self._cnn_prediction_montage_image()
        self._add_image_to_doc(document, montage, Inches(5.8), "CNN - Prediction montage")

    def _export_cnn_ppt(self, prs, subtab_name: str, Inches) -> None:
        tab = self.cnn_tab

        settings = [
            (
                "Dataset: "
                f"size={int(tab.image_size_var.get())}, samples/class={int(tab.samples_per_class_var.get())}, "
                f"split={float(tab.test_split_var.get()):.2f}, seed={int(tab.seed_var.get())}"
            ),
            (
                "Augmentation: "
                f"rotation={'on' if tab.apply_rotation_var.get() else 'off'} ({float(tab.rotation_range_var.get()):.1f}), "
                f"noise={'on' if tab.apply_noise_var.get() else 'off'} (std={float(tab.noise_std_var.get()):.1f})"
            ),
            (
                "Training: "
                f"mode={tab.mode_var.get()}, epochs={int(tab.epochs_var.get())}, "
                f"batch={int(tab.batch_size_var.get())}, tf_logs={tab.tf_log_level_var.get()}"
            ),
        ]
        if tab.metrics:
            settings.append(
                f"Evaluation: accuracy={tab.metrics['accuracy'] * 100:.2f}% | loss={tab.metrics['loss']:.4f}"
            )
        else:
            settings.append("Evaluation: not available yet (run Train + Evaluate in CNN tab)")

        self._ppt_add_summary_slide(prs, f"{subtab_name} - Summary", settings)

        history_stream = self._cnn_history_figure_png()
        if history_stream is not None:
            self._ppt_add_stream_slide(prs, subtab_name, "CNN - Training curves", history_stream, Inches)
        else:
            self._ppt_add_summary_slide(prs, subtab_name, ["CNN - Training curves not available"])

        montage = self._cnn_prediction_montage_image()
        self._ppt_add_image_slide(prs, subtab_name, "CNN - Prediction montage", montage, Inches)

    def _cnn_history_figure_png(self) -> io.BytesIO | None:
        history = self.cnn_tab.history
        if not history:
            return None

        fig = Figure(figsize=(7.0, 2.8), dpi=120, facecolor="#ffffff")
        ax = fig.add_subplot(111)
        ax.set_title("CNN Training Curves", fontsize=10)
        ax.grid(alpha=0.25)

        epochs = np.arange(1, len(history.get("loss", [])) + 1)
        if len(epochs) > 0:
            ax.plot(epochs, history.get("loss", []), color="#d62728", linewidth=1.2, label="Train Loss")
            ax.plot(epochs, history.get("val_loss", []), color="#ff7f0e", linewidth=1.2, label="Val Loss")
            ax.plot(epochs, history.get("accuracy", []), color="#1f77b4", linewidth=1.2, label="Train Acc")
            ax.plot(epochs, history.get("val_accuracy", []), color="#2ca02c", linewidth=1.2, label="Val Acc")
            ax.set_xlim(1, max(1, int(epochs[-1])))
            ax.legend(fontsize=8)

        stream = io.BytesIO()
        fig.tight_layout()
        fig.savefig(stream, format="png")
        stream.seek(0)
        return stream

    def _cnn_prediction_montage_image(self) -> np.ndarray | None:
        tab = self.cnn_tab
        if (
            tab.prediction_x is None
            or tab.prediction_true is None
            or tab.prediction_ids is None
            or tab.prediction_conf is None
        ):
            return None

        tile_size = 120
        cols = 4
        total = int(tab.prediction_x.shape[0])
        rows = int(np.ceil(total / cols))
        canvas = np.full((rows * tile_size, cols * tile_size, 3), 24, dtype=np.uint8)

        for idx in range(total):
            row = idx // cols
            col = idx % cols
            y0 = row * tile_size
            x0 = col * tile_size

            gray = np.clip(tab.prediction_x[idx].squeeze() * 255.0, 0, 255).astype(np.uint8)
            tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)

            true_id = int(tab.prediction_true[idx])
            pred_id = int(tab.prediction_ids[idx])
            conf = float(tab.prediction_conf[idx])
            is_correct = true_id == pred_id
            color = (80, 220, 140) if is_correct else (220, 120, 120)

            cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), color, 2)
            cv2.putText(
                tile,
                f"T:{'C' if true_id == 0 else 'S'} P:{'C' if pred_id == 0 else 'S'}",
                (6, tile_size - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                tile,
                f"C:{conf:.2f}",
                (6, tile_size - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

            canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile

        return canvas

    def _add_image_to_doc(self, document, image: np.ndarray | None, width, caption: str) -> None:
        if image is None:
            document.add_paragraph(f"{caption}: not available")
            return

        success, encoded = cv2.imencode(".png", image)
        if not success:
            document.add_paragraph(f"{caption}: failed to encode")
            return
        stream = io.BytesIO(encoded.tobytes())
        self._add_stream_figure(document, stream, width, caption)

    def _add_stream_figure(self, document, stream: io.BytesIO, width, caption: str) -> None:
        document.add_picture(stream, width=width)
        self._export_figure_index += 1
        document.add_paragraph(f"Figure {self._export_figure_index}: {caption}")

    @staticmethod
    def _add_treeview_table_to_doc(document, tree, title: str) -> None:
        columns = list(tree["columns"])
        rows = [tree.item(item_id).get("values", []) for item_id in tree.get_children()]

        document.add_paragraph(title)
        if not rows:
            document.add_paragraph("No rows available.")
            return

        table = document.add_table(rows=len(rows) + 1, cols=len(columns))
        table.style = "Table Grid"

        for col_idx, col_name in enumerate(columns):
            table.cell(0, col_idx).text = str(col_name)

        for row_idx, row_values in enumerate(rows, start=1):
            for col_idx, value in enumerate(row_values):
                table.cell(row_idx, col_idx).text = str(value)

    @staticmethod
    def _hist_figure_png(image: np.ndarray, title: str) -> io.BytesIO:
        fig = Figure(figsize=(7.0, 2.8), dpi=120, facecolor="#ffffff")
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=10)
        ax.set_xlim([0, 256])
        ax.grid(alpha=0.25)

        hist_data = compute_histogram(image)
        if "gray" in hist_data:
            ax.plot(hist_data["gray"], color="black", linewidth=1.2)
            ax.legend(["Gray"], fontsize=8)
        else:
            ax.plot(hist_data["b"], color="blue", linewidth=1.1)
            ax.plot(hist_data["g"], color="green", linewidth=1.1)
            ax.plot(hist_data["r"], color="red", linewidth=1.1)
            ax.legend(["Blue", "Green", "Red"], fontsize=8)

        stream = io.BytesIO()
        fig.tight_layout()
        fig.savefig(stream, format="png")
        stream.seek(0)
        return stream
