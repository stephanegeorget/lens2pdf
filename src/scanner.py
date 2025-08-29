"""Document scanning and OCR using a CZUR lens camera."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import math

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .camera import CameraInfo, list_cameras, select_camera
from .image_utils import (
    correct_orientation,
    find_document_contour,
    four_point_transform,
    increase_contrast,
    reduce_jpeg_artifacts,
)
from . import ocr_utils

# Re-export modules for tests to monkeypatch
shutil = ocr_utils.shutil
Path = ocr_utils.Path
pytesseract = ocr_utils.pytesseract
Image = ocr_utils.Image

# Scale factor for preview windows (e.g. 0.5 = half size)
PREVIEW_SCALE = 0.5

# Cache an opened camera so subsequent scans can reuse the stream without
# re-enumerating available devices which can take a long time.
_cached_cap: cv2.VideoCapture | None = None


def _debug_time(start: float, label: str) -> float:
    """Print a debug message showing elapsed time since ``start``."""
    now = time.perf_counter()
    print(f"[DEBUG] {label}: {now - start:.2f}s")
    return now


def _create_window(name: str) -> None:
    """Create an OpenCV window and bring it to the foreground.

    ``cv2.setWindowProperty`` with ``WND_PROP_TOPMOST`` is best-effort and
    silently ignored if the current platform or OpenCV build does not support
    it.  This ensures the preview window is visible instead of hiding behind
    other applications.
    """

    cv2.namedWindow(name)
    try:  # pragma: no cover - depends on GUI backend
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass


def _is_v_sign(hand) -> bool:
    """Return ``True`` if the hand landmarks form a ``V`` gesture.

    The gesture is detected based on the distance of finger tips from the
    wrist which makes the check orientation agnostic. This allows triggering
    the scan even when the camera is rotated 90 degrees and the ``V`` appears
    as ``>`` or ``<``.
    """

    lm = hand.landmark
    wrist = lm[0]

    def dist(a, b):
        return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

    def extended(tip, pip):
        return dist(lm[tip], wrist) > dist(lm[pip], wrist)

    def folded(tip, pip):
        return dist(lm[tip], wrist) < dist(lm[pip], wrist)

    # Index and middle fingers extended
    if not (extended(8, 6) and extended(12, 10)):
        return False
    # Ring and pinky fingers folded
    if not (folded(16, 14) and folded(20, 18)):
        return False
    # Tips reasonably far apart to form a V shape
    tip_separation = math.hypot(lm[8].x - lm[12].x, lm[8].y - lm[12].y)
    if tip_separation < 0.1:
        return False
    return True


def check_tesseract_installation() -> None:  # pragma: no cover - thin wrapper
    """Proxy to ``ocr_utils.check_tesseract_installation`` using local modules."""
    ocr_utils.shutil = shutil
    ocr_utils.Path = Path
    ocr_utils.pytesseract = pytesseract
    return ocr_utils.check_tesseract_installation()


def save_pdf(image: np.ndarray, output_dir: Path | str | None = None):  # pragma: no cover - thin wrapper
    """Proxy to ``ocr_utils.save_pdf`` using local modules."""
    ocr_utils.shutil = shutil
    ocr_utils.Path = Path
    ocr_utils.pytesseract = pytesseract
    return ocr_utils.save_pdf(image, output_dir)


def open_pdf(path: Path) -> None:  # pragma: no cover - OS-specific side effect
    """Open ``path`` using the platform's default PDF viewer."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform.startswith("darwin"):
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"Unable to open PDF {path}: {exc}")


def test_camera() -> None:
    """Display the camera feed without scanning to verify the window."""
    start = time.perf_counter()
    print("[DEBUG] Starting test_camera")
    cameras = list_cameras()
    _debug_time(start, "after list_cameras")
    cam_index = select_camera(cameras)
    _debug_time(start, "after select_camera")
    cap = cv2.VideoCapture(cam_index)
    _debug_time(start, "after VideoCapture")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
    _debug_time(start, "after setting resolution")
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    _debug_time(start, "after cap.isOpened")
    _create_window("Camera Test")
    _debug_time(start, "after namedWindow")
    print("Press 'q' to quit.")
    first_frame = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame:
            _debug_time(start, "after first frame")
            first_frame = False
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


def scan_document(
    skip_detection: bool = False,
    gesture_enabled: bool = True,
    boost_contrast: bool = True,
    output_dir: Path | str | None = None,
    timeout: float | None = None,
    *,
    min_area_ratio: float = 0.1,
) -> bool:
    """Run the interactive document scanner.

    Parameters
    ----------
    timeout:
        Maximum number of seconds to wait for a scan request before exiting.
        ``None`` disables the timeout.
    min_area_ratio:
        Forwarded to :func:`src.image_utils.find_document_contour` to control the
        minimum size of detectable documents.
    Returns
    -------
    bool
        ``True`` if a document was scanned, ``False`` otherwise.
    """
    start = time.perf_counter()
    print("[DEBUG] Starting scan_document")

    global _cached_cap
    if _cached_cap is None:
        cameras = list_cameras()
        _debug_time(start, "after list_cameras")
        cam_index = select_camera(cameras)
        _debug_time(start, "after select_camera")
        _cached_cap = cv2.VideoCapture(cam_index)
        _debug_time(start, "after VideoCapture")
        _cached_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
        _cached_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
        _debug_time(start, "after setting resolution")
        if not _cached_cap.isOpened():
            raise RuntimeError("Unable to open camera")
        _debug_time(start, "after cap.isOpened")
    cap = _cached_cap

    _create_window("Scanner")
    _debug_time(start, "after namedWindow")
    print("Press 's' to scan or 'q' to quit.")

    # Hand gesture detector
    hands = None
    if gesture_enabled:
        try:
            import mediapipe as mp  # type: ignore

            hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
        except Exception:  # pragma: no cover - mediapipe optional
            print("mediapipe not available, disabling gesture trigger")
            gesture_enabled = False

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
    first_frame = True
    wait_start = time.monotonic()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame:
            _debug_time(start, "after first frame")
            first_frame = False
        display = frame.copy()
        if not skip_detection:
            contour = find_document_contour(
                frame, min_area_ratio=min_area_ratio, preview=display
            )

        if gesture_enabled and hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks and any(
                _is_v_sign(h) for h in results.multi_hand_landmarks
            ):
                for i in (3, 2, 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    display = frame.copy()
                    text = str(i)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 16
                    thickness = 12
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, font, font_scale, thickness
                    )
                    x = (display.shape[1] - text_width) // 2
                    y = (display.shape[0] + text_height) // 2
                    cv2.putText(
                        display,
                        text,
                        (x, y),
                        font,
                        font_scale,
                        (0, 255, 0),
                        thickness,
                        cv2.LINE_AA,
                    )
                    if PREVIEW_SCALE != 1.0:
                        display = cv2.resize(
                            display,
                            (0, 0),
                            fx=PREVIEW_SCALE,
                            fy=PREVIEW_SCALE,
                            interpolation=cv2.INTER_AREA,
                        )
                    cv2.imshow("Scanner", display)
                    cv2.waitKey(1000)
                ret, frame = cap.read()
                break

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

        if timeout is not None and time.monotonic() - wait_start > timeout:
            frame = None
            break

    if frame is None:
        cap.release()
        _cached_cap = None
    cv2.destroyAllWindows()
    if frame is None:
        return False

    if not skip_detection:
        contour = find_document_contour(frame, min_area_ratio=min_area_ratio)
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
    if boost_contrast:
        corrected = increase_contrast(corrected)
    # Lightly denoise the frame to reduce visible JPEG artifacts before saving
    corrected = reduce_jpeg_artifacts(corrected)
    pdf_path = save_pdf(corrected, output_dir)
    print(f"Saved {pdf_path}")
    open_pdf(pdf_path)

    return True


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
    parser.add_argument(
        "--no-gesture",
        action="store_true",
        help="disable hand gesture scan trigger",
    )
    parser.add_argument(
        "--no-contrast",
        action="store_true",
        help="disable 25% contrast boost",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory to store generated PDF files",
    )
    args = parser.parse_args()
    if args.test_camera:
        test_camera()
    else:
        while scan_document(
            skip_detection=args.no_detect,
            gesture_enabled=not args.no_gesture,
            boost_contrast=not args.no_contrast,
            output_dir=args.output_dir,
            timeout=60,
        ):
            pass


if __name__ == "__main__":
    main()
