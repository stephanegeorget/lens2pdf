"""Image processing helpers for document scanning."""

from __future__ import annotations

import os
import re

import cv2
import numpy as np
import pytesseract

from .ocr_utils import check_tesseract_installation


def find_document_contour(frame: np.ndarray) -> np.ndarray | None:
    """Locate a rectangular contour in ``frame``."""
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
        if len(approx) == 4 and area > 0.5 * frame_area:
            return approx
    if contours:
        area = cv2.contourArea(contours[0])
        if area > 0.9 * frame_area:
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


def rotate_bound(image: np.ndarray, angle: int) -> np.ndarray:
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


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """Rotate ``image`` based on Tesseract's orientation detection."""
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


__all__ = [
    "find_document_contour",
    "order_points",
    "four_point_transform",
    "rotate_bound",
    "correct_orientation",
]
