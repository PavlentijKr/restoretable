from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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
    alignment_tolerance: int = 4
    poppler_path: Optional[str] = None

    def horizontal_kernel(self, width: int) -> np.ndarray:
        length = max(10, width // self.min_kernel_scale)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

    def vertical_kernel(self, height: int) -> np.ndarray:
        length = max(10, height // self.min_kernel_scale)
        return cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))


@dataclass
class LineSegment:
    """Detected line segment from the binary mask."""

    orientation: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int

    @property
    def coordinate(self) -> int:
        return self.y_start if self.orientation == "horizontal" else self.x_start


@dataclass
class TableLine:
    """Reconstructed continuous table line."""

    orientation: str
    start: Tuple[int, int]
    end: Tuple[int, int]

    @property
    def coordinate(self) -> int:
        return self.start[1] if self.orientation == "horizontal" else self.start[0]


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

    def _separate_orientations(self, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    ) -> List["LineSegment"]:
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
        self, segments: Sequence["LineSegment"], tolerance: int
    ) -> List[List["LineSegment"]]:
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

    def _build_lines_from_groups(
        self, groups: Sequence[Sequence["LineSegment"]], orientation: str
    ) -> List["TableLine"]:
        lines: List[TableLine] = []
        for group in groups:
            if orientation == "horizontal":
                y_coord = int(round(sum(seg.coordinate for seg in group) / len(group)))
                x_min = min(seg.x_start for seg in group)
                x_max = max(seg.x_end for seg in group)
                lines.append(TableLine("horizontal", (x_min, y_coord), (x_max, y_coord)))
            else:
                x_coord = int(round(sum(seg.coordinate for seg in group) / len(group)))
                y_min = min(seg.y_start for seg in group)
                y_max = max(seg.y_end for seg in group)
                lines.append(TableLine("vertical", (x_coord, y_min), (x_coord, y_max)))
        return lines

    def _fill_missing_lines(
        self, lines: List["TableLine"], orientation: str, width: int, height: int
    ) -> List["TableLine"]:
        if len(lines) < 2:
            return lines

        sorted_lines = sorted(lines, key=lambda line: line.coordinate)
        if orientation == "horizontal":
            global_start = min(line.start[0] for line in sorted_lines)
            global_end = max(line.end[0] for line in sorted_lines)
        else:
            global_start = min(line.start[1] for line in sorted_lines)
            global_end = max(line.end[1] for line in sorted_lines)

        if orientation == "horizontal" and global_start == global_end:
            global_start, global_end = 0, width - 1
        if orientation == "vertical" and global_start == global_end:
            global_start, global_end = 0, height - 1

        changed = True
        while changed:
            changed = False
            coordinates = [line.coordinate for line in sorted_lines]
            gaps = [b - a for a, b in zip(coordinates, coordinates[1:])]
            if not gaps:
                break
            median_gap = float(np.median(gaps))
            if median_gap == 0:
                break
            for idx, gap in enumerate(gaps):
                if gap > 1.5 * median_gap:
                    new_coord = int(round((coordinates[idx] + coordinates[idx + 1]) / 2))
                    if orientation == "horizontal":
                        new_line = TableLine(
                            "horizontal", (global_start, new_coord), (global_end, new_coord)
                        )
                    else:
                        new_line = TableLine(
                            "vertical", (new_coord, global_start), (new_coord, global_end)
                        )
                    sorted_lines.insert(idx + 1, new_line)
                    changed = True
                    break
        return sorted_lines

    def _reconstruct_lines(
        self, binary: np.ndarray
    ) -> Tuple[List["TableLine"], List["TableLine"]]:
        horizontal_mask, vertical_mask = self._separate_orientations(binary)
        height, width = binary.shape

        horizontal_min_len = max(5, width // self.config.min_kernel_scale)
        vertical_min_len = max(5, height // self.config.min_kernel_scale)

        horizontal_segments = self._extract_segments(
            horizontal_mask, "horizontal", horizontal_min_len
        )
        vertical_segments = self._extract_segments(vertical_mask, "vertical", vertical_min_len)

        tolerance = max(1, self.config.alignment_tolerance)
        horizontal_groups = self._group_segments(horizontal_segments, tolerance)
        vertical_groups = self._group_segments(vertical_segments, tolerance)

        horizontal_lines = self._build_lines_from_groups(horizontal_groups, "horizontal")
        vertical_lines = self._build_lines_from_groups(vertical_groups, "vertical")

        horizontal_lines = self._fill_missing_lines(horizontal_lines, "horizontal", width, height)
        vertical_lines = self._fill_missing_lines(vertical_lines, "vertical", width, height)

        return horizontal_lines, vertical_lines

    def _draw_lines(
        self, horizontal_lines: Sequence["TableLine"], vertical_lines: Sequence["TableLine"], shape: Tuple[int, int]
    ) -> np.ndarray:
        height, width = shape
        line_mask = np.zeros((height, width), dtype=np.uint8)
        thickness = max(1, self.config.line_thickness)
        for line in list(horizontal_lines) + list(vertical_lines):
            cv2.line(line_mask, line.start, line.end, 255, thickness)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(
            line_mask, cv2.MORPH_CLOSE, close_kernel, iterations=self.config.gap_close_iterations
        )

    def _overlay_lines(self, original: np.ndarray, lines: np.ndarray) -> np.ndarray:
        output = original.copy()
        output[lines == 255] = [0, 0, 0]
        return output

    def restore_tables_on_image(self, image: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        binary = self._adaptive_binary(bgr)
        horizontal_lines, vertical_lines = self._reconstruct_lines(binary)
        line_mask = self._draw_lines(horizontal_lines, vertical_lines, binary.shape)
        restored_bgr = self._overlay_lines(bgr, line_mask)
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
