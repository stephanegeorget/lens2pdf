"""Camera discovery and selection utilities."""

from __future__ import annotations

import os
import re
import sys
import time
import io
import subprocess
from dataclasses import dataclass
from typing import List

import cv2

# Windows-specific helpers
if sys.platform == "win32":
    import msvcrt

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

    if sys.platform == "win32":
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "dshow",
            "-list_devices",
            "true",
            "-i",
            "dummy",
        ]
    elif sys.platform == "darwin":
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "avfoundation",
            "-list_devices",
            "true",
            "-i",
            "",
        ]
    else:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-f",
            "v4l2",
            "-list_devices",
            "true",
            "-i",
            "dummy",
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stderr
    except Exception as exc:  # pragma: no cover - ffmpeg may be missing
        print(f"Unable to run ffmpeg: {exc}")
        output = ""

    current_name = None
    for line in output.splitlines():
        alt = re.search(r'Alternative name\s+"([^"]+)"', line)
        if alt and current_name:
            cameras.append(
                CameraInfo(
                    index=len(cameras),
                    name=current_name,
                    backend="FFMPEG",
                    hw_address=alt.group(1),
                )
            )
            current_name = None
            continue

        m = re.search(r'\]\s+"([^"]+)"$', line)
        if m:
            current_name = m.group(1)

    if not cameras:
        print("No cameras detected by ffmpeg")
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
