"""PyInstaller-safe launcher with an absolute package import."""

from contour_preprocess_tool.app import main


if __name__ == "__main__":
    raise SystemExit(main())
