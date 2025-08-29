"""Document scanning and OCR using a CZUR lens camera."""

from __future__ import annotations

import argparse
import re
import sys
import threading
import queue
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract

CAMERA_REGEX = re.compile(r"czur\s+lens", re.IGNORECASE)


def list_cameras(max_devices: int = 5) -> list[tuple[int, str]]:
    """Return a list of available camera indices and names."""
    cameras: list[tuple[int, str]] = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            name = f"Camera {index}"
            if hasattr(cv2, "CAP_PROP_DEVICE_DESCRIPTION"):
                try:
                    # Some platforms expose the device name via this property
                    prop = cap.get(cv2.CAP_PROP_DEVICE_DESCRIPTION)
                    if isinstance(prop, str) and prop:
                        name = prop
                except Exception:  # pragma: no cover - best effort
                    pass
            cameras.append((index, name))
        cap.release()
    return cameras


def select_camera(cameras: list[tuple[int, str]]) -> int:
    """Select a camera from ``cameras``.

    If a camera name matches ``CAMERA_REGEX`` it is selected by default.
    The user has two seconds to enter another index before the default is used.
    """
    if not cameras:
        raise RuntimeError("No cameras found")

    default = cameras[0][0]
    for idx, name in cameras:
        if CAMERA_REGEX.search(name):
            default = idx
            break

    print("Available cameras:")
    for idx, name in cameras:
        label = "(default)" if idx == default else ""
        print(f"[{idx}] {name} {label}")
    print("Press Enter within 2 seconds to select another camera index.")

    q: queue.Queue[str] = queue.Queue()

    def reader() -> None:
        q.put(input())

    t = threading.Thread(target=reader)
    t.daemon = True
    t.start()
    try:
        choice = q.get(timeout=2).strip()
        if choice.isdigit():
            return int(choice)
    except queue.Empty:
        pass
    return default


def find_document_contour(frame: np.ndarray) -> np.ndarray | None:
    """Locate a rectangular contour in ``frame``."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
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
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """Rotate ``image`` based on Tesseract's orientation detection."""
    osd = pytesseract.image_to_osd(image)
    match = re.search(r"Rotate: (\d+)", osd)
    angle = int(match.group(1)) if match else 0
    if angle:
        image = rotate_bound(image, angle)
    return image


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


def save_pdf(image: np.ndarray) -> Path:
    """Save ``image`` with OCR text as a timestamped PDF file."""
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension="pdf")
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".pdf"
    path = Path(filename)
    path.write_bytes(pdf_bytes)
    return path


def test_camera() -> None:
    """Display the camera feed without scanning to verify the window."""
    cameras = list_cameras()
    cam_index = select_camera(cameras)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    cv2.namedWindow("Camera Test")
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def scan_document() -> None:
    """Run the interactive document scanner."""
    cameras = list_cameras()
    cam_index = select_camera(cameras)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    cv2.namedWindow("Scanner")
    print("Press 's' to scan or 'q' to quit.")

    # Read single characters from stdin so the user can exit even if the
    # OpenCV window fails to appear or loses focus.  This mirrors the
    # behaviour of ``cv2.waitKey`` which captures key presses in the window.
    stdin_q: queue.Queue[str] = queue.Queue()

    def stdin_reader() -> None:
        while True:  # pragma: no cover - best effort for interactive use
            ch = sys.stdin.read(1)
            if not ch:
                break
            stdin_q.put(ch)

    threading.Thread(target=stdin_reader, daemon=True).start()

    frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        contour = find_document_contour(frame)
        display = frame.copy()
        if contour is not None:
            cv2.polylines(display, [contour], True, (0, 255, 0), 2)
        cv2.imshow("Scanner", display)

        # Gather key presses from the OpenCV window and the terminal.
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
    contour = find_document_contour(frame)
    if contour is None:
        print("No document detected.")
        return
    warped = four_point_transform(frame, contour)
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
    args = parser.parse_args()
    if args.test_camera:
        test_camera()
    else:
        scan_document()


if __name__ == "__main__":
    main()
