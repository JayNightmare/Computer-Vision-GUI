"""
GUI Entry Point
===============
Launches the Image Processing Dashboard.

Usage:
    python gui_main.py
"""

from gui import ImageProcessorApp


def main() -> None:
    app = ImageProcessorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
