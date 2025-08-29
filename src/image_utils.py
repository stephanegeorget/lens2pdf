"""Image processing helpers for document scanning."""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
import pytesseract

from .ocr_utils import check_tesseract_installation


def find_document_contour(
    frame: np.ndarray, *, min_area_ratio: float = 0.1
) -> np.ndarray | None:
    """Locate a rectangular contour in ``frame``.

    Parameters
    ----------
    frame:
        Image in which to search for the document.
    min_area_ratio:
        Minimum area (as a fraction of the full frame) that a contour must
        cover to be considered a document.  Smaller values allow detection of
        documents that do not fill most of the camera view.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = frame.shape[:2]
    frame_area = width * height
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > min_area_ratio * frame_area:
            return approx
    if contours:
        area = cv2.contourArea(contours[0])
        if area > max(min_area_ratio, 0.9) * frame_area:
            return np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )
    return None


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return a consistent ordering of the four points of ``pts``."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perform a perspective transform of ``image`` using ``pts``."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``image`` by ``angle`` degrees without cropping."""
    (h, w) = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    m[0, 2] += (n_w / 2) - w / 2
    m[1, 2] += (n_h / 2) - h / 2
    return cv2.warpAffine(image, m, (n_w, n_h))


def detect_dominant_edge_angle(image: np.ndarray) -> float:
    """Return the angle of the strongest edge in ``image`` in degrees.

    The image is converted to grayscale, edges are detected using Canny, and a
    Hough transform is applied.  The line with the highest vote count (``cv2.HoughLines``)
    or, if none are detected, the longest probabilistic line (``cv2.HoughLinesP``)
    is used.  The returned angle represents the deviation from the vertical
    axis; a perfectly upright edge yields ``0``.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    angle = 0.0
    if lines is not None:
        rho, theta = lines[0][0]
        angle = np.degrees(theta) - 90
    else:
        lines_p = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=10
        )
        if lines_p is not None:
            x1, y1, x2, y2 = max(
                lines_p,
                key=lambda l: np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]),
            )[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) - 90
    # Normalize to [-90, 90)
    angle = (angle + 90) % 180 - 90
    return angle


def correct_orientation(image: np.ndarray, contour: np.ndarray | None = None) -> np.ndarray:
    """Rotate ``image`` using edge detection and Tesseract orientation.

    If ``contour`` is ``None`` (indicating document contour detection failed),
    the function first estimates the dominant edge angle and rotates the image
    to make that edge upright.  It then performs an OCR-based orientation
    detection to correct any remaining 90° multiples.
    """

    if contour is None:
        theta = detect_dominant_edge_angle(image)
        if abs(theta) > 0.1:
            image = rotate_bound(image, theta)

    check_tesseract_installation()
    try:
        osd = pytesseract.image_to_osd(image)
        match = re.search(r"Rotate: (\d+)", osd)
        angle = int(match.group(1)) if match else 0
    except Exception:
        angle = 0
    if angle in (90, 180, 270):
        image = rotate_bound(image, angle)
    return image


def increase_contrast(image: np.ndarray, factor: float = 1.25) -> np.ndarray:
    """Return ``image`` with its contrast scaled by ``factor``."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


__all__ = [
    "find_document_contour",
    "order_points",
    "four_point_transform",
    "rotate_bound",
    "detect_dominant_edge_angle",
    "correct_orientation",
    "increase_contrast",
]
