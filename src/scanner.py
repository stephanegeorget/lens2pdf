"""Document scanning and OCR for overhead cameras such as the CZUR Lens.

The scanner highlights page edges and boosts contrast but does not correct
perspective, so position the camera parallel to the document.
"""

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
    find_long_edges,
    increase_contrast,
    reduce_jpeg_artifacts,
    correct_orientation,
)
from . import ocr_utils, __version__

# Re-export modules for tests to monkeypatch
shutil = ocr_utils.shutil
Path = ocr_utils.Path
pytesseract = ocr_utils.pytesseract
Image = ocr_utils.Image

# Scale factor for preview windows. ``0.5`` would be half size, ``1.0`` would
# show the full camera resolution.  Using a quarter-sized preview keeps the
# on-screen window compact even when the capture resolution is high.
PREVIEW_SCALE = 0.5

# Scale factor applied before running heavy OpenCV algorithms.  Processing a
# smaller image reduces CPU usage while the original resolution is kept for
# the final capture.
PROCESSING_SCALE = 1.0

# Default capture resolution chosen to balance quality and responsiveness.  A
# lower resolution keeps the preview smooth while still providing enough detail
# for OCR.
CAPTURE_WIDTH = 2592 # max 3264
CAPTURE_HEIGHT = 1944 # max 2448

# Highest resolution supported by the camera when capturing a still image.
FULL_RES_WIDTH = 3264
FULL_RES_HEIGHT = 2448

# Resolution used for a smooth live preview when ``fast_preview`` is enabled.
VIDEO_PREVIEW_WIDTH = 1920
VIDEO_PREVIEW_HEIGHT = 1080

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


def _stack_frames(cap: cv2.VideoCapture, base: np.ndarray, count: int) -> np.ndarray:
    """Return the average of ``count`` frames from ``cap``.

    The first frame to include in the average is provided via ``base``; the
    function then grabs ``count - 1`` additional frames from ``cap`` and
    computes their arithmetic mean.  This simple stacking technique reduces
    noise and can recover detail when the document being scanned is stationary.
    """

    acc = base.astype(np.float32)
    frames = 1
    for _ in range(count - 1):
        ret, frame = cap.read()
        if not ret:
            break
        acc += frame.astype(np.float32)
        frames += 1
    acc /= frames
    return np.clip(acc, 0, 255).astype(np.uint8)


def _open_capture(cam_index: int, cameras) -> cv2.VideoCapture:
    """Open ``cam_index`` using the backend associated with ``cameras``.

    ``cameras`` may contain :class:`CameraInfo` objects or simple ``(index,
    name)`` tuples.  If a backend string is available, this function attempts
    to map it to the corresponding OpenCV constant so that the same backend is
    used for enumeration and streaming.  This avoids situations where the
    device order differs between backends (e.g. DirectShow vs Media Foundation
    on Windows).
    """

    cam = None
    for c in cameras:
        idx = getattr(c, "index", None)
        if idx is None and isinstance(c, tuple):
            idx = c[0]
        if idx == cam_index:
            cam = c
            break

    backend_const = None
    backend_name = getattr(cam, "backend", None) if cam is not None else None
    if backend_name:
        const_name = (
            backend_name if backend_name.startswith("CAP_") else f"CAP_{backend_name}"
        )
        backend_const = getattr(cv2, const_name, None)

    if backend_const is not None:
        return cv2.VideoCapture(cam_index, backend_const)
    return cv2.VideoCapture(cam_index)


def check_tesseract_installation() -> None:  # pragma: no cover - thin wrapper
    """Proxy to ``ocr_utils.check_tesseract_installation`` using local modules."""
    ocr_utils.shutil = shutil
    ocr_utils.Path = Path
    ocr_utils.pytesseract = pytesseract
    return ocr_utils.check_tesseract_installation()


def save_pdf(
    image: np.ndarray, output_dir: Path | str | None = None
):  # pragma: no cover - thin wrapper
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
    cap = _open_capture(cam_index, cameras)
    _debug_time(start, "after VideoCapture")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
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
        key = cv2.waitKey(1)
        # Allow quitting either by pressing ``q`` or closing the window
        if (key != -1 and chr(key & 0xFF).lower() == "q") or cv2.getWindowProperty(
            "Camera Test", cv2.WND_PROP_VISIBLE
        ) < 1:
            break
    cap.release()
    cv2.destroyAllWindows()


def scan_document(
    gesture_enabled: bool = True,
    boost_contrast: bool = True,
    output_dir: Path | str | None = None,
    timeout: float | None = None,
    stack_count: int = 10,
    angle_threshold: float = 2,
    fast_preview: bool = False,
) -> bool:
    """Run the interactive document scanner.

    Parameters
    ----------
    gesture_enabled:
        Enable hand gesture detection using MediaPipe. When disabled a scan is
        triggered only by pressing ``s``.
    boost_contrast:
        Apply a mild contrast stretch prior to OCR to improve legibility.
    output_dir:
        Optional directory in which to save generated PDF files.
    timeout:
        Maximum number of seconds to wait for a scan request before exiting.
        ``None`` disables the timeout.
    stack_count:
        Number of frames to average together for a single scan. Using multiple
        frames can reduce noise and slightly improve effective resolution when
        the document is stationary.
    angle_threshold:
        Maximum allowed deviation in degrees for edges to be drawn green in the
        preview.
    fast_preview:
        Use a video-friendly resolution and backend for the live preview and
        switch to the highest still-image resolution when capturing.

    Returns
    -------
    bool
        ``True`` if a document was scanned, ``False`` otherwise.
    """
    start = time.perf_counter()
    print("[DEBUG] Starting scan_document")
    global _cached_cap
    cameras = None
    cam_index = None
    if fast_preview:
        cameras = list_cameras()
        _debug_time(start, "after list_cameras")
        cam_index = select_camera(cameras)
        _debug_time(start, "after select_camera")
        cap = cv2.VideoCapture(cam_index, getattr(cv2, "CAP_MSMF", 0))
        _debug_time(start, "after VideoCapture")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_PREVIEW_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_PREVIEW_HEIGHT)
        _debug_time(start, "after setting resolution")
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera")
        _debug_time(start, "after cap.isOpened")
    else:
        if _cached_cap is None:
            cameras = list_cameras()
            _debug_time(start, "after list_cameras")
            cam_index = select_camera(cameras)
            _debug_time(start, "after select_camera")
            _cached_cap = _open_capture(cam_index, cameras)
            _debug_time(start, "after VideoCapture")
            _cached_cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            _cached_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
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
        """Collect characters typed in the terminal.

        OpenCV's highgui window only receives key events when it has focus; the
        background thread allows triggering scans from the terminal as well.
        """

        while True:
            ch = sys.stdin.read(1)
            if not ch:
                break
            stdin_q.put(ch)

    threading.Thread(target=stdin_reader, daemon=True).start()

    frame = None
    capture_triggered = False
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

        # Downscale the frame for CPU-intensive processing steps.
        proc = frame
        if PROCESSING_SCALE != 1.0:
            interp = getattr(cv2, "INTER_AREA", None)
            if interp is not None:
                proc = cv2.resize(
                    frame,
                    (0, 0),
                    fx=PROCESSING_SCALE,
                    fy=PROCESSING_SCALE,
                    interpolation=interp,
                )
            else:  # pragma: no cover - fallback when constant missing
                proc = cv2.resize(
                    frame,
                    (0, 0),
                    fx=PROCESSING_SCALE,
                    fy=PROCESSING_SCALE,
                )

        # Show prominent edges so the user can adjust the document alignment.
        edges = find_long_edges(proc)
        scale = 1.0 / PROCESSING_SCALE if PROCESSING_SCALE != 1.0 else 1.0
        for x1, y1, x2, y2, angle in edges:
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
            diff = min(abs(angle), abs(angle - 90))
            color = (0, 255, 0) if diff <= angle_threshold else (0, 0, 255)
            cv2.line(display, (x1, y1), (x2, y2), color, 2)

        if gesture_enabled and hands is not None:
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks and any(
                _is_v_sign(h) for h in results.multi_hand_landmarks
            ):
                for i in (4, 3, 2, 1):
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
                capture_triggered = True
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

        key = cv2.waitKey(1)
        key_char = ""
        if key != -1:
            key_char = chr(key & 0xFF).lower()
        while not stdin_q.empty():
            key_char = stdin_q.get_nowait().lower()

        if key_char == "s" or key == 13:
            capture_triggered = True
            break
        if (
            key_char == "q"
            or cv2.getWindowProperty("Scanner", cv2.WND_PROP_VISIBLE) < 1
        ):
            frame = None
            break

        if timeout is not None and time.monotonic() - wait_start > timeout:
            frame = None
            break

    if fast_preview:
        cap.release()
        if not capture_triggered:
            cv2.destroyAllWindows()
            return False
        cap = _open_capture(cam_index, cameras)
        _debug_time(start, "after reopen for capture")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FULL_RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FULL_RES_HEIGHT)
        _debug_time(start, "after setting full resolution")
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            return False
        if stack_count > 1:
            frame = _stack_frames(cap, frame, stack_count)
        cap.release()
    else:
        if not capture_triggered:
            cap.release()
            _cached_cap = None
            cv2.destroyAllWindows()
            return False
        if frame is not None and stack_count > 1:
            frame = _stack_frames(cap, frame, stack_count)

    cv2.destroyAllWindows()

    if frame is None:
        return False

    frame = correct_orientation(frame)
    if boost_contrast:
        frame = increase_contrast(frame)
    # Lightly denoise the frame to reduce visible JPEG artifacts before saving
    frame = reduce_jpeg_artifacts(frame)

    # Save the image as a searchable PDF and open it with the default viewer.
    pdf_path = save_pdf(frame, output_dir)
    print(f"Saved {pdf_path}")
    open_pdf(pdf_path)

    return True


def build_parser() -> argparse.ArgumentParser:
    """Return the ``argparse`` configuration for the command line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Turn an overhead webcam into a document scanner. Perspective is "
            "not corrected, so keep the camera parallel to the page."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show program's version number and exit",
    )
    parser.add_argument(
        "--test-camera",
        action="store_true",
        help="display raw camera feed and exit; useful for diagnostics",
    )
    parser.add_argument(
        "--no-gesture",
        action="store_true",
        help="disable MediaPipe V-sign detection; use the 's' key to scan",
    )
    parser.add_argument(
        "--no-contrast",
        action="store_true",
        help="skip the 25%% contrast boost before performing OCR",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory in which to write PDF files",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=2,
        help="maximum deviation in degrees for edges to appear green",
    )
    parser.add_argument(
        "--stack-count",
        type=int,
        default=10,
        help=(
            "number of frames to average for each capture; higher values "
            "reduce noise at the cost of speed"
        ),
    )
    parser.add_argument(
        "--fast-preview",
        action="store_true",
        help=(
            "use a smooth MSMF preview and switch to full resolution for the "
            "actual scan"
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="after saving a PDF, start a new scan instead of exiting",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the scanner script."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test_camera:
        test_camera()
    else:
        if args.loop:
            while scan_document(
                gesture_enabled=not args.no_gesture,
                boost_contrast=not args.no_contrast,
                output_dir=args.output_dir,
                timeout=60,
                stack_count=args.stack_count,
                angle_threshold=args.angle_threshold,
                fast_preview=args.fast_preview,
            ):
                pass
        else:
            scan_document(
                gesture_enabled=not args.no_gesture,
                boost_contrast=not args.no_contrast,
                output_dir=args.output_dir,
                timeout=60,
                stack_count=args.stack_count,
                angle_threshold=args.angle_threshold,
                fast_preview=args.fast_preview,
            )


if __name__ == "__main__":
    main()
