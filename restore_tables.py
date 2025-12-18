from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


@dataclass
class TableRestorationConfig:
    """Configuration for table restoration."""

    dpi: int = 300
    line_thickness: int = 2
    adaptive_block_size: int = 35
    adaptive_c: int = 10
    min_kernel_scale: int = 50
    gap_close_iterations: int = 2
    poppler_path: Optional[str] = None

    def horizontal_kernel(self, width: int) -> np.ndarray:
        length = max(10, width // self.min_kernel_scale)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

    def vertical_kernel(self, height: int) -> np.ndarray:
        length = max(10, height // self.min_kernel_scale)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))


class TableRestorer:
    def __init__(self, config: Optional[TableRestorationConfig] = None) -> None:
        self.config = config or TableRestorationConfig()

    def pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        return convert_from_path(
            str(pdf_path),
            dpi=self.config.dpi,
            poppler_path=self.config.poppler_path,
        )

    def _adaptive_binary(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )
        return binary

    def _extract_grid(self, binary: np.ndarray) -> np.ndarray:
        height, width = binary.shape
        horizontal_kernel = self.config.horizontal_kernel(width)
        vertical_kernel = self.config.vertical_kernel(height)

        horizontal = cv2.erode(binary, horizontal_kernel, iterations=1)
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=2)

        vertical = cv2.erode(binary, vertical_kernel, iterations=1)
        vertical = cv2.dilate(vertical, vertical_kernel, iterations=2)

        grid = cv2.add(horizontal, vertical)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grid = cv2.morphologyEx(
            grid, cv2.MORPH_CLOSE, close_kernel, iterations=self.config.gap_close_iterations
        )

        return grid

    def _thicken_lines(self, grid: np.ndarray) -> np.ndarray:
        thickness = max(1, self.config.line_thickness)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
        return cv2.dilate(grid, kernel, iterations=1)

    def _overlay_lines(self, original: np.ndarray, lines: np.ndarray) -> np.ndarray:
        output = original.copy()
        output[lines == 255] = [0, 0, 0]
        return output

    def restore_tables_on_image(self, image: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        binary = self._adaptive_binary(bgr)
        grid = self._extract_grid(binary)
        thick_lines = self._thicken_lines(grid)
        restored_bgr = self._overlay_lines(bgr, thick_lines)
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(restored_rgb)

    def process_pdf(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        pages = self.pdf_to_images(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: List[Path] = []

        for idx, page in enumerate(pages, start=1):
            restored = self.restore_tables_on_image(page)
            output_path = output_dir / f"page-{idx}.png"
            restored.save(output_path)
            output_paths.append(output_path)

        return output_paths


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore table lines in PDF pages.")
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Directory to store processed PNG pages",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Thickness (in pixels) for the restored grid lines",
    )
    parser.add_argument(
        "--poppler-path",
        type=str,
        default=None,
        help="Optional path to Poppler binaries (if not on PATH)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config = TableRestorationConfig(
        dpi=args.dpi,
        line_thickness=args.line_thickness,
        poppler_path=args.poppler_path,
    )
    restorer = TableRestorer(config)

    output_paths = restorer.process_pdf(args.pdf, args.output)
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
