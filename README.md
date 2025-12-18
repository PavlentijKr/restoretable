# restoretable

Utility for converting PDFs to PNG images and restoring table grids by thickening and reconnecting lines.

## Setup

1. Install Python 3.9+.
2. (Recommended) Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `pdf2image` requires Poppler. On macOS, install via Homebrew (`brew install poppler`); on Ubuntu/Debian, use `sudo apt-get install poppler-utils`.

## Usage

Process a PDF and write enhanced PNG pages to `./output`:

```bash
python restore_tables.py input.pdf
```

Choose a custom output directory, DPI, and line thickness:

```bash
python restore_tables.py input.pdf --output processed_pages --dpi 300 --line-thickness 3
```

If Poppler is not on your `PATH`, provide its location:

```bash
python restore_tables.py input.pdf --poppler-path /usr/local/opt/poppler/bin
```

Each page is saved as `page-<n>.png` in the output directory.

## How it works

1. Convert PDF pages to images with `pdf2image`.
2. Convert to grayscale and apply adaptive thresholding to isolate ink.
3. Split the binary mask into horizontal and vertical candidates using oriented morphology kernels.
4. Extract connected components as line segments, group them by alignment, and reconstruct continuous horizontal and vertical lines with averaged coordinates and unified extents.
5. Detect missing grid rows/columns using the median spacing heuristic and synthesize intermediate lines.
6. Draw the reconstructed full-length lines at a uniform thickness and overlay them on the original image without altering text.
