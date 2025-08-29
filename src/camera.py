"""Camera discovery and selection utilities."""

from __future__ import annotations

import os
import re
import sys
import time
import io
from dataclasses import dataclass
from typing import List

import cv2

# Windows-specific helpers
FilterGraph = None
if sys.platform == "win32":
    import msvcrt

    try:
        from pygrabber.dshow_graph import FilterGraph as _FG

        FilterGraph = _FG
    except ImportError:
        print("Warning: pygrabber not installed, camera names may not be accurate.")

# Match any CZUR-branded device regardless of model suffix.
CAMERA_REGEX = re.compile(r"czur", re.IGNORECASE)


@dataclass(eq=False)
class CameraInfo:
    """Debug information about a discovered camera device."""

    index: int
    name: str
    backend: str | None = None
    description: str | None = None
    hw_address: str | None = None

    def summary(self) -> str:
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


def timed_input(prompt: str, timeout: int = 2) -> str | None:
    """Read user input with a timeout. Works on Windows and POSIX."""
    print(prompt, end="", flush=True)

    if sys.platform == "win32":
        start = time.time()
        buf = ""
        while True:
            if msvcrt.kbhit():
                char = msvcrt.getwche()
                if char in ("\r", "\n"):
                    print()
                    return buf
                if char == "\b":
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
            return input().strip()


def list_cameras(max_devices: int = 5) -> List[CameraInfo]:
    """Return a list of available camera indices and debug information."""
    print(f"Python: {sys.version.split()[0]} ({sys.platform})")
    print(f"OpenCV version: {getattr(cv2, '__version__', 'unknown')}")
    cameras: list[CameraInfo] = []

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

    print("Probing camera indices...")
    for index in range(max_devices):
        if sys.platform == "win32" and hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            name = f"Camera {index}"
            if sys.platform == "win32" and index < len(win_names):
                name = win_names[index]
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
            cap.release()
            break
        cap.release()
    return cameras


def select_camera(cameras: List[CameraInfo]) -> int:
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


__all__ = [
    "CameraInfo",
    "list_cameras",
    "select_camera",
    "timed_input",
]
