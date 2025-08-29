"""Document scanning and OCR using a CZUR lens camera."""

from __future__ import annotations

import argparse
import queue
import sys
import threading

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .camera import CameraInfo, list_cameras, select_camera
from .image_utils import (
    correct_orientation,
    find_document_contour,
    four_point_transform,
)
from . import ocr_utils

# Re-export modules for tests to monkeypatch
shutil = ocr_utils.shutil
Path = ocr_utils.Path
pytesseract = ocr_utils.pytesseract
Image = ocr_utils.Image

# Scale factor for preview windows (e.g. 0.5 = half size)
PREVIEW_SCALE = 0.5


def check_tesseract_installation() -> None:  # pragma: no cover - thin wrapper
    """Proxy to ``ocr_utils.check_tesseract_installation`` using local modules."""
    ocr_utils.shutil = shutil
    ocr_utils.Path = Path
    ocr_utils.pytesseract = pytesseract
    return ocr_utils.check_tesseract_installation()


def save_pdf(image: np.ndarray):  # pragma: no cover - thin wrapper
    """Proxy to ``ocr_utils.save_pdf`` using local modules."""
    ocr_utils.shutil = shutil
    ocr_utils.Path = Path
    ocr_utils.pytesseract = pytesseract
    return ocr_utils.save_pdf(image)


def test_camera() -> None:
    """Display the camera feed without scanning to verify the window."""
    cameras = list_cameras()
    cam_index = select_camera(cameras)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    cv2.namedWindow("Camera Test")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preview = frame
        if PREVIEW_SCALE != 1.0:
            preview = cv2.resize(
                frame,
                (0, 0),
                fx=PREVIEW_SCALE,
                fy=PREVIEW_SCALE,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow("Camera Test", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def scan_document(skip_detection: bool = False) -> None:
    """Run the interactive document scanner."""
    cameras = list_cameras()
    cam_index = select_camera(cameras)
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    cv2.namedWindow("Scanner")
    print("Press 's' to scan or 'q' to quit.")

    stdin_q: queue.Queue[str] = queue.Queue()

    def stdin_reader() -> None:
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                break
            stdin_q.put(ch)

    threading.Thread(target=stdin_reader, daemon=True).start()

    frame = None
    contour = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        if not skip_detection:
            contour = find_document_contour(frame)
            if contour is not None:
                cv2.polylines(display, [contour], True, (0, 255, 0), 2)
        if PREVIEW_SCALE != 1.0:
            display = cv2.resize(
                display,
                (0, 0),
                fx=PREVIEW_SCALE,
                fy=PREVIEW_SCALE,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow("Scanner", display)

        key = cv2.waitKey(1) & 0xFF
        while not stdin_q.empty():
            char = stdin_q.get_nowait().lower()
            if char == "q":
                key = ord("q")
            elif char == "s":
                key = ord("s")

        if key in (ord("s"), 13):
            break
        if key == ord("q"):
            frame = None
            break

    cap.release()
    cv2.destroyAllWindows()
    if frame is None:
        return

    if not skip_detection:
        contour = find_document_contour(frame)
    else:
        contour = None
    if contour is not None and not skip_detection:
        warped = four_point_transform(frame, contour)
    else:
        warped = frame
    if skip_detection:
        corrected = warped
    else:
        corrected = correct_orientation(warped)
    pdf_path = save_pdf(corrected)
    print(f"Saved {pdf_path}")


def main() -> None:
    """Entry point for the scanner script."""
    parser = argparse.ArgumentParser(description="Document scanner")
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="display camera feed without scanning",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="disable document bounding box and rotation detection",
    )
    args = parser.parse_args()
    if args.test_camera:
        test_camera()
    else:
        scan_document(skip_detection=args.no_detect)


if __name__ == "__main__":
    main()
