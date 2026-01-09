from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


@dataclass
class MultiTypeTSRConfig:
    """Configuration for multi-type table structure recognition."""

    dpi: int = 300
    adaptive_block_size: int = 35
    adaptive_c: int = 10
    min_kernel_scale: int = 60
    alignment_tolerance: int = 4
    min_line_count: int = 2
    projection_smooth: int = 9
    projection_percentile: float = 70.0
    min_band_size: int = 10
    min_cell_size: int = 8
    poppler_path: Optional[str] = None

    def horizontal_kernel(self, width: int) -> np.ndarray:
        length = max(12, width // self.min_kernel_scale)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

    def vertical_kernel(self, height: int) -> np.ndarray:
        length = max(12, height // self.min_kernel_scale)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))


@dataclass
class LineSegment:
    orientation: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    @property
    def coordinate(self) -> int:
        return self.y_start if self.orientation == "horizontal" else self.x_start


class MultiTypeTableStructureRecognizer:
    def __init__(self, config: Optional[MultiTypeTSRConfig] = None) -> None:
        self.config = config or MultiTypeTSRConfig()

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

    def _separate_lines(self, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = binary.shape
        horizontal_kernel = self.config.horizontal_kernel(width)
        vertical_kernel = self.config.vertical_kernel(height)

        horizontal = cv2.erode(binary, horizontal_kernel, iterations=1)
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=1)

        vertical = cv2.erode(binary, vertical_kernel, iterations=1)
        vertical = cv2.dilate(vertical, vertical_kernel, iterations=1)

        return horizontal, vertical

    def _extract_segments(
        self, mask: np.ndarray, orientation: str, min_length: int
    ) -> List[LineSegment]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments: List[LineSegment] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if orientation == "horizontal":
                if w < min_length or w < h:
                    continue
                y_coord = y + h // 2
                segments.append(LineSegment("horizontal", x, y_coord, x + w, y_coord))
            else:
                if h < min_length or h < w:
                    continue
                x_coord = x + w // 2
                segments.append(LineSegment("vertical", x_coord, y, x_coord, y + h))
        return segments

    def _group_segments(
        self, segments: Sequence[LineSegment], tolerance: int
    ) -> List[List[LineSegment]]:
        if not segments:
            return []
        sorted_segments = sorted(segments, key=lambda seg: seg.coordinate)
        groups: List[List[LineSegment]] = [[sorted_segments[0]]]
        for seg in sorted_segments[1:]:
            last_group = groups[-1]
            group_coord = sum(s.coordinate for s in last_group) / len(last_group)
            if abs(seg.coordinate - group_coord) <= tolerance:
                last_group.append(seg)
            else:
                groups.append([seg])
        return groups

    def _group_to_coordinates(self, groups: Sequence[Sequence[LineSegment]]) -> List[int]:
        coords = []
        for group in groups:
            coord = int(round(sum(seg.coordinate for seg in group) / len(group)))
            coords.append(coord)
        return sorted(coords)

    def _table_bounds(self, binary: np.ndarray) -> Tuple[int, int, int, int]:
        points = cv2.findNonZero(binary)
        height, width = binary.shape
        if points is None:
            return 0, 0, width - 1, height - 1
        x, y, w, h = cv2.boundingRect(points)
        return x, y, x + w - 1, y + h - 1

    def _smooth_projection(self, projection: np.ndarray) -> np.ndarray:
        kernel_size = max(3, self.config.projection_smooth)
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(projection, kernel, mode="same")

    def _bands_from_projection(self, projection: np.ndarray) -> List[Tuple[int, int]]:
        threshold = np.percentile(projection, self.config.projection_percentile)
        active = projection >= threshold
        bands: List[Tuple[int, int]] = []
        start = None
        for idx, value in enumerate(active):
            if value and start is None:
                start = idx
            if not value and start is not None:
                if idx - start >= self.config.min_band_size:
                    bands.append((start, idx - 1))
                start = None
        if start is not None and len(active) - start >= self.config.min_band_size:
            bands.append((start, len(active) - 1))
        return bands

    def _centers_to_boundaries(
        self, centers: Sequence[int], start: int, end: int
    ) -> List[int]:
        if not centers:
            return [start, end]
        sorted_centers = sorted(centers)
        boundaries = [start]
        for left, right in zip(sorted_centers, sorted_centers[1:]):
            boundaries.append(int(round((left + right) / 2)))
        boundaries.append(end)
        return boundaries

    def _borderless_grid(self, binary: np.ndarray) -> Tuple[List[int], List[int]]:
        height, width = binary.shape
        row_proj = self._smooth_projection(binary.sum(axis=1))
        col_proj = self._smooth_projection(binary.sum(axis=0))

        row_bands = self._bands_from_projection(row_proj)
        col_bands = self._bands_from_projection(col_proj)

        row_centers = [int(round((start + end) / 2)) for start, end in row_bands]
        col_centers = [int(round((start + end) / 2)) for start, end in col_bands]

        x_min, y_min, x_max, y_max = self._table_bounds(binary)
        row_boundaries = self._centers_to_boundaries(row_centers, y_min, y_max)
        col_boundaries = self._centers_to_boundaries(col_centers, x_min, x_max)
        return row_boundaries, col_boundaries

    def _grid_from_lines(
        self, binary: np.ndarray
    ) -> Tuple[List[int], List[int], bool]:
        horizontal_mask, vertical_mask = self._separate_lines(binary)
        height, width = binary.shape
        min_h_len = max(10, width // self.config.min_kernel_scale)
        min_v_len = max(10, height // self.config.min_kernel_scale)
        horizontal_segments = self._extract_segments(horizontal_mask, "horizontal", min_h_len)
        vertical_segments = self._extract_segments(vertical_mask, "vertical", min_v_len)
        tolerance = max(1, self.config.alignment_tolerance)
        horizontal_groups = self._group_segments(horizontal_segments, tolerance)
        vertical_groups = self._group_segments(vertical_segments, tolerance)
        horizontal_coords = self._group_to_coordinates(horizontal_groups)
        vertical_coords = self._group_to_coordinates(vertical_groups)

        if (
            len(horizontal_coords) >= self.config.min_line_count
            and len(vertical_coords) >= self.config.min_line_count
        ):
            x_min, y_min, x_max, y_max = self._table_bounds(binary)
            row_boundaries = [y_min] + horizontal_coords + [y_max]
            col_boundaries = [x_min] + vertical_coords + [x_max]
            return sorted(set(row_boundaries)), sorted(set(col_boundaries)), True

        row_boundaries, col_boundaries = self._borderless_grid(binary)
        return row_boundaries, col_boundaries, False

    def _cells_from_boundaries(
        self, rows: Sequence[int], cols: Sequence[int]
    ) -> List[dict]:
        cells: List[dict] = []
        for row_idx, (y_start, y_end) in enumerate(zip(rows, rows[1:])):
            for col_idx, (x_start, x_end) in enumerate(zip(cols, cols[1:])):
                if x_end - x_start < self.config.min_cell_size:
                    continue
                if y_end - y_start < self.config.min_cell_size:
                    continue
                cells.append(
                    {
                        "row": row_idx,
                        "col": col_idx,
                        "bbox": [int(x_start), int(y_start), int(x_end), int(y_end)],
                    }
                )
        return cells

    def detect_table_structure(self, image: Image.Image) -> dict:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        binary = self._adaptive_binary(bgr)
        row_boundaries, col_boundaries, is_grid = self._grid_from_lines(binary)
        cells = self._cells_from_boundaries(row_boundaries, col_boundaries)
        mode = "grid" if is_grid else "borderless"
        return {
            "mode": mode,
            "rows": row_boundaries,
            "cols": col_boundaries,
            "cells": cells,
        }

    def annotate_cells(self, image: Image.Image, cells: Sequence[dict]) -> Image.Image:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for cell in cells:
            x_start, y_start, x_end, y_end = cell["bbox"]
            cv2.rectangle(bgr, (x_start, y_start), (x_end, y_end), (0, 165, 255), 2)
        annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated)

    def process_pdf(self, pdf_path: Path, output_dir: Path, save_annotated: bool) -> List[Path]:
        pages = self.pdf_to_images(pdf_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: List[Path] = []

        for idx, page in enumerate(pages, start=1):
            result = self.detect_table_structure(page)
            json_path = output_dir / f"page-{idx}.json"
            json_path.write_text(json.dumps(result, indent=2))
            output_paths.append(json_path)

            if save_annotated:
                annotated = self.annotate_cells(page, result["cells"])
                image_path = output_dir / f"page-{idx}.png"
                annotated.save(image_path)
                output_paths.append(image_path)

        return output_paths

    def process_image(self, image_path: Path, output_dir: Path, save_annotated: bool) -> List[Path]:
        image = Image.open(image_path).convert("RGB")
        result = self.detect_table_structure(image)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{image_path.stem}.json"
        json_path.write_text(json.dumps(result, indent=2))
        output_paths = [json_path]

        if save_annotated:
            annotated = self.annotate_cells(image, result["cells"])
            image_path = output_dir / f"{image_path.stem}.png"
            annotated.save(image_path)
            output_paths.append(image_path)

        return output_paths


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-friendly multi-type table structure recognition.",
    )
    parser.add_argument("input", type=Path, help="Path to a PDF or image file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tsr_output"),
        help="Directory to store JSON and annotated images",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated images with detected cells",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)",
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
    config = MultiTypeTSRConfig(dpi=args.dpi, poppler_path=args.poppler_path)
    recognizer = MultiTypeTableStructureRecognizer(config)

    if args.input.suffix.lower() == ".pdf":
        output_paths = recognizer.process_pdf(args.input, args.output, args.save_annotated)
    else:
        output_paths = recognizer.process_image(args.input, args.output, args.save_annotated)

    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
