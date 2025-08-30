"""Image processing helpers for document scanning."""

from __future__ import annotations

import cv2
import numpy as np
import re
import pytesseract

from .ocr_utils import check_tesseract_installation


def find_long_edges(
    image: np.ndarray,
    *,
    min_length_ratio: float = 0.25,
    max_edges: int = 20,
) -> list[tuple[int, int, int, int, float]]:
    """Return long edges detected in ``image``.

    Edges are detected using Canny edge detection followed by a probabilistic
    Hough transform.  Only line segments longer than ``min_length_ratio`` times
    the smaller image dimension are returned.  The segments are sorted by
    length from longest to shortest and limited to ``max_edges`` entries.

    Returns a list of ``(x1, y1, x2, y2, angle)`` tuples where ``angle`` is the
    absolute angle of the line segment in degrees.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    h, w = gray.shape[:2]
    min_len = int(min(h, w) * min_length_ratio)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=min_len, maxLineGap=10
    )

    results: list[tuple[int, int, int, int, float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle > 180:
                angle -= 180
            results.append((x1, y1, x2, y2, angle, length))
        results.sort(key=lambda x: x[5], reverse=True)

    return [(x1, y1, x2, y2, angle) for x1, y1, x2, y2, angle, _ in results[:max_edges]]


def increase_contrast(image: np.ndarray, factor: float = 1.25) -> np.ndarray:
    """Return ``image`` with its contrast scaled by ``factor``."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def reduce_jpeg_artifacts(image: np.ndarray) -> np.ndarray:
    """Denoise ``image`` to lessen visible JPEG compression artifacts."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """Rotate ``image`` to its upright orientation using Tesseract OSD."""
    check_tesseract_installation()
    try:
        osd = pytesseract.image_to_osd(image)
        match = re.search(r"Rotate: (\d+)", osd)
        angle = int(match.group(1)) if match else 0
    except Exception:
        angle = 0

    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


__all__ = [
    "find_long_edges",
    "increase_contrast",
    "reduce_jpeg_artifacts",
    "correct_orientation",
]
