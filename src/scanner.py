"""Document scanning and OCR using a CZUR lens camera."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import queue
from dataclasses import dataclass
from types import SimpleNamespace
import time
from datetime import datetime
from pathlib import Path
import shutil
import io

import cv2
import numpy as np
import pytesseract
from PIL import Image

# ------------------------------------------------------------
# Windows-specific helpers
# ------------------------------------------------------------
FilterGraph = None  # placeholder, set below if available

if sys.platform == "win32":
    import msvcrt
    try:
        from pygrabber.dshow_graph import FilterGraph as _FG
        FilterGraph = _FG
    except ImportError:
        print("Warning: pygrabber not installed, camera names may not be accurate.")
        # pip install pygrabber

# ------------------------------------------------------------
# Data classes
# ------------------------------------------------------------
@dataclass(eq=False)
class CameraInfo:
    """Debug information about a discovered camera device."""

    index: int
    name: str
    backend: str | None = None
    description: str | None = None
    hw_address: str | None = None

    def summary(self) -> str:
        """Return a human readable summary of the camera."""
        extras: list[str] = []
        if self.backend:
            extras.append(f"backend={self.backend}")
        if self.hw_address:
            extras.append(f"hw={self.hw_address}")
        if self.description and self.description != self.name:
            extras.append(f"desc={self.description}")
        return " ".join(extras)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return (self.index, self.name) == other
        if isinstance(other, CameraInfo):
            return (
                self.index == other.index
                and self.name == other.name
                and self.backend == other.backend
                and self.description == other.description
                and self.hw_address == other.hw_address
            )
        return NotImplemented


# Match any CZUR-branded device regardless of model suffix.
CAMERA_REGEX = re.compile(r"czur", re.IGNORECASE)

# Scale factor for preview windows (e.g. 0.5 = half size)
PREVIEW_SCALE = 0.5


# ------------------------------------------------------------
# Input helper
# ------------------------------------------------------------
def timed_input(prompt: str, timeout: int = 2) -> str | None:
    """Read user input with a timeout. Works on Windows and POSIX."""
    print(prompt, end="", flush=True)

    if sys.platform == "win32":
        start = time.time()
        buf = ""
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ("\r", "\n"):  # Enter
                    print()
                    return buf
                elif char == "\b":  # Backspace
                    buf = buf[:-1]
                    print("\b \b", end="", flush=True)
                else:
                    buf += char
            if time.time() - start > timeout:
                print()
                return None
            time.sleep(0.05)
    else:
        import select
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.readline().strip()
            return None
        except (OSError, io.UnsupportedOperation):
            # Fallback when stdin does not provide a fileno (e.g., tests)
            return input().strip()


# ------------------------------------------------------------
# Camera handling
# ------------------------------------------------------------
def list_cameras(max_devices: int = 5) -> list[CameraInfo]:
    """Return a list of available camera indices and debug information."""
    print(f"Python: {sys.version.split()[0]} ({sys.platform})")
    print(f"OpenCV version: {getattr(cv2, '__version__', 'unknown')}")
    cameras: list[CameraInfo] = []

    # Windows: get friendly device names if pygrabber is available
    win_names: list[str] = []
    if sys.platform == "win32" and FilterGraph is not None:
        try:
            graph = FilterGraph()
            win_names = graph.get_input_devices()
        except Exception:
            pass

    registry = getattr(cv2, "videoio_registry", None)
    if registry and hasattr(registry, "getCameraInfoList"):
        try:
            infos = registry.getCameraInfoList()
            backend = None
            if hasattr(registry, "getBackends") and hasattr(registry, "getBackendName"):
                backends = registry.getBackends()
                if backends:
                    try:
                        backend = registry.getBackendName(backends[0])
                    except Exception:
                        backend = None
            for info in infos:
                cameras.append(
                    CameraInfo(
                        index=getattr(info, "id", len(cameras)),
                        name=getattr(info, "name", f"Camera {len(cameras)}"),
                        backend=str(backend) if backend else None,
                    )
                )
        except Exception:
            pass
    if cameras:
        return cameras

    # Probe indices
    print("Probing camera indices...")
    for index in range(max_devices):
        if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            name = f"Camera {index}"
            if sys.platform == "win32" and index < len(win_names):
                name = win_names[index]  # use friendly name if available
            backend = None
            try:
                backend = cap.getBackendName()  # type: ignore[attr-defined]
            except Exception:
                pass
            cameras.append(
                CameraInfo(
                    index=index,
                    name=name,
                    backend=str(backend) if backend else None,
                )
            )
            print(f"  index {index}: name={name} backend={backend}")
        else:
            # Camera indices are typically contiguous; stop after the first
            # missing index to avoid long timeouts when probing.
            cap.release()
            break
        cap.release()
    return cameras


def select_camera(cameras: list[CameraInfo]) -> int:
    """Select a camera from ``cameras``."""
    if not cameras:
        raise RuntimeError("No cameras found")

    default = cameras[0].index
    for cam in cameras:
        if CAMERA_REGEX.search(cam.name):
            default = cam.index
            break

    print("Available cameras:")
    for cam in cameras:
        label = "(default)" if cam.index == default else ""
        extras = cam.summary()
        extra_str = f" {extras}" if extras else ""
        print(f"[{cam.index}] {cam.name} {label}{extra_str}")

    choice = timed_input(
        f"Enter camera index within 2 seconds (default={default}): ", timeout=2
    )
    if choice and choice.isdigit():
        return int(choice)
    return default


# ------------------------------------------------------------
# OCR / Tesseract
# ------------------------------------------------------------
def check_tesseract_installation() -> None:
    """Ensure that the Tesseract executable is available."""
    cmd = shutil.which("tesseract")
    if cmd:
        return

    win_path = Path("C:/pf/Tesseract-OCR/tesseract.exe")
    if win_path.is_file():
        if not hasattr(pytesseract, "pytesseract"):
            pytesseract.pytesseract = SimpleNamespace()
        pytesseract.pytesseract.tesseract_cmd = str(win_path)
        return

    raise RuntimeError(
        "Tesseract OCR is required. Install it from "
        "https://github.com/UB-Mannheim/tesseract/wiki and ensure it "
        "is installed in C:\\pf\\Tesseract-OCR."
    )


# ------------------------------------------------------------
# Document contour detection
# ------------------------------------------------------------
def find_document_contour(frame: np.ndarray) -> np.ndarray | None:
    """Locate a rectangular contour in ``frame``."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, m, (max_width, max_height))


def correct_orientation(image: np.ndarray) -> np.ndarray:
    """Rotate ``image`` based on Tesseract's orientation detection (90° steps)."""
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
    """Save ``image`` with OCR text as a high-resolution PDF file."""
    check_tesseract_installation()

    # Convert to PIL Image and set DPI
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img.info["dpi"] = (300, 300)

    pdf_bytes = pytesseract.image_to_pdf_or_hocr(pil_img, extension="pdf")
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".pdf"
    path = Path(filename)
    path.write_bytes(pdf_bytes)
    return path


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def test_camera() -> None:
    """Display the camera feed without scanning to verify the window."""
    cameras = list_cameras()
    cam_index = select_camera(cameras)
    cap = cv2.VideoCapture(cam_index)

    # Set high resolution
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


def scan_document() -> None:
    """Run the interactive document scanner."""
    cameras = list_cameras()
    cam_index = select_camera(cameras)
    cap = cv2.VideoCapture(cam_index)

    # Set high resolution
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        contour = find_document_contour(frame)
        display = frame.copy()
        if contour is not None:
            cv2.polylines(display, [contour], True, (0, 255, 0), 2)
        preview = display
        if PREVIEW_SCALE != 1.0:
            preview = cv2.resize(
                display,
                (0, 0),
                fx=PREVIEW_SCALE,
                fy=PREVIEW_SCALE,
                interpolation=cv2.INTER_AREA,
            )
        cv2.imshow("Scanner", preview)

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
    if contour is not None:
        warped = four_point_transform(frame, contour)
    else:
        print("No document detected; using full frame.")
        warped = frame
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
