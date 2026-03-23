# Project Memory — Workshops 3 & 4: Image Processing

## Project Overview

Computer Vision Workshops 3 & 4 — Retinex image enhancement (CLI),
interactive point-processing GUI dashboard (Workshop 3), and spatial /
statistical filtering GUI (Workshop 4).

## Architecture

```
Workshop 2/
├── theory/retinex_theory.md       # Theory write-up (no code)
├── retinex/retinex.py             # Retinex algorithm (SSR + MSR)
├── processing/
│   ├── point_ops.py               # Workshop 3: Point processing (7 functions)
│   └── spatial_ops.py             # Workshop 4: Spatial/statistical (13 functions)
├── gui/
│   ├── app.py                     # Tk root + nested notebooks (WS3 / WS4)
│   ├── dashboard_tab.py           # WS3: Image preview + sliders + histograms
│   ├── histogram_tab.py           # WS3: CLAHE + Negative comparison
│   ├── arithmetic_tab.py          # WS4: Pixel-wise add/subtract/multiply
│   ├── edge_detection_tab.py      # WS4: Sobel/Scharr/Laplacian/Roberts/Prewitt
│   └── filtering_tab.py           # WS4: Noise + Average/Median filter + MSE
├── utils/visualization.py         # CLI comparison figures
├── main.py                        # CLI entry point (Retinex)
├── gui_main.py                    # GUI entry point
├── image/                         # Input images
└── output/                        # CLI-generated results
```

## Completed Tasks

- [x] Retinex theory, SSR/MSR algorithm, CLI runner, verification
- [x] GUI dashboard with dark theme (Catppuccin-inspired)
  - Dashboard tab: original/altered preview, histograms, sliders, equalize/reset
  - Histogram tab: Current / CLAHE / Negative side-by-side
- [x] Workshop 4: Spatial & Statistical Filtering
  - Arithmetic tab: load two images, add/subtract/multiply, 3-panel display
  - Edge Detection tab: 5 operators (Sobel, Scharr, Laplacian, Roberts, Prewitt) side-by-side
  - Filtering tab: Gaussian/S&P noise, Average/Median filter, MSE readout
- [x] Restructured notebook into Workshop 3 / Workshop 4 top-level tab groups

## Key Decisions

- **Separation of concerns**: Algorithms in `processing/` (point_ops + spatial_ops), no UI code.
  GUI reads from those modules. Retinex algorithm remains in its own `retinex/` package.
- **Tkinter** chosen — no extra frameworks needed; ships with Python.
- **Virtual environment**: deps in `venv/` — numpy, opencv-python, matplotlib, Pillow.
- **Dark theme**: Catppuccin Mocha palette via `ttk.Style` overrides.
- **Nested notebooks**: Outer tabs for workshops, inner tabs for features. Distinct styling
  (blue accent outer, grey accent inner) to visually separate the hierarchy.

## Next Steps

- Could add image save/export from the GUI.
- Could add MSR/SSR as an additional tab or processing option.
- Could add frequency-domain filtering in a future workshop tab.
