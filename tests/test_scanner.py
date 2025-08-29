from types import SimpleNamespace
import importlib
import sys
from pathlib import Path
import pytest



def setup_fake_cv2(monkeypatch):
    """Return the scanner module loaded with a stubbed cv2 module."""

    class FakeCapture:
        def __init__(self, index):
            self.index = index

        def isOpened(self):
            return False

        def release(self):
            pass

    info1 = SimpleNamespace(id=0, name="Generic Cam")
    info2 = SimpleNamespace(id=1, name="CZUR E012A")

    registry = SimpleNamespace(
        getCameraInfoList=lambda: [info1, info2],
        getBackends=lambda: [1],
        getBackendName=lambda _b: "FAKE",
    )
    fake_cv2 = SimpleNamespace(
        videoio_registry=registry,
        VideoCapture=FakeCapture,
        __version__="4.8.0",
    )
    # ``scanner`` imports ``cv2``, ``numpy`` and ``pytesseract`` at module import
    # time.  Provide lightweight stubs for these modules so the import succeeds
    # without requiring the heavy dependencies.
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pytesseract", SimpleNamespace())

    import src.scanner as scanner
    importlib.reload(scanner)
    return scanner


def test_list_cameras_uses_device_names(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    cams = scanner.list_cameras()

    assert [(c.index, c.name) for c in cams] == [
        (0, "Generic Cam"),
        (1, "CZUR E012A"),
    ]


def test_select_camera_defaults_to_czur(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda: "")
    cams = scanner.list_cameras()
    assert scanner.select_camera(cams) == 1

    assert cams == [(0, "Generic Cam"), (1, "CZUR E012A")]


def test_check_tesseract_missing(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    monkeypatch.setattr(scanner.shutil, "which", lambda cmd: None)

    class FakePath:
        def __init__(self, path):
            self.path = path

        def is_file(self):
            return False

        def __str__(self):
            return self.path

    monkeypatch.setattr(scanner, "Path", FakePath)
    with pytest.raises(RuntimeError):
        scanner.check_tesseract_installation()


def test_check_tesseract_configures_path(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    monkeypatch.setattr(scanner.shutil, "which", lambda cmd: None)

    class FakePath:
        def __init__(self, path):
            self.path = path

        def is_file(self):
            return True

        def __str__(self):
            return self.path

    monkeypatch.setattr(scanner, "Path", FakePath)
    scanner.check_tesseract_installation()
    assert (
        scanner.pytesseract.pytesseract.tesseract_cmd
        == "C:/pf/Tesseract-OCR/tesseract.exe"
    )


def test_no_gesture_flag(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    called = {}

    def fake_scan(*, skip_detection, gesture_enabled, boost_contrast, output_dir):
        called["args"] = (skip_detection, gesture_enabled, boost_contrast, output_dir)

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(sys, "argv", ["scanner", "--no-gesture"])
    scanner.main()

    assert called["args"] == (False, False, True, None)


def test_no_contrast_flag(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    called = {}

    def fake_scan(*, skip_detection, gesture_enabled, boost_contrast, output_dir):
        called["args"] = (skip_detection, gesture_enabled, boost_contrast, output_dir)

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(sys, "argv", ["scanner", "--no-contrast"])
    scanner.main()

    assert called["args"] == (False, True, False, None)


def test_output_dir_flag(monkeypatch, tmp_path):
    scanner = setup_fake_cv2(monkeypatch)
    called = {}

    def fake_scan(*, skip_detection, gesture_enabled, boost_contrast, output_dir):
        called["args"] = (skip_detection, gesture_enabled, boost_contrast, output_dir)

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scanner", "--output-dir", str(tmp_path)],
    )
    scanner.main()

    assert called["args"] == (False, True, True, str(tmp_path))


def test_is_v_sign_sideways(monkeypatch):
    """The V gesture should be detected even when rotated 90 degrees."""
    scanner = setup_fake_cv2(monkeypatch)

    def lm(x, y):
        return SimpleNamespace(x=x, y=y)

    def build_hand(mapping):
        lms = [lm(0, 0) for _ in range(21)]
        for idx, (x, y) in mapping.items():
            lms[idx] = lm(x, y)
        return SimpleNamespace(landmark=lms)

    # ``>`` shape (camera rotated clockwise)
    right_hand = build_hand(
        {
            0: (0, 0),
            6: (0.4, 0.4),
            8: (0.8, 0.4),
            10: (0.4, 0.6),
            12: (0.8, 0.6),
            14: (0.4, 0.7),
            16: (0.2, 0.7),
            18: (0.4, 0.8),
            20: (0.2, 0.8),
        }
    )
    assert scanner._is_v_sign(right_hand)

    # ``<`` shape (camera rotated counter-clockwise)
    left_hand = build_hand(
        {
            0: (1, 0),
            6: (0.6, 0.4),
            8: (0.2, 0.4),
            10: (0.6, 0.6),
            12: (0.2, 0.6),
            14: (0.6, 0.7),
            16: (0.8, 0.7),
            18: (0.6, 0.8),
            20: (0.8, 0.8),
        }
    )
    assert scanner._is_v_sign(left_hand)


def test_open_pdf_linux(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    opened = {}

    def fake_run(cmd, check):
        opened["cmd"] = cmd

    monkeypatch.setattr(scanner.subprocess, "run", fake_run)
    monkeypatch.setattr(scanner.sys, "platform", "linux")
    scanner.open_pdf(Path("doc.pdf"))

    assert opened["cmd"] == ["xdg-open", "doc.pdf"]


def test_open_pdf_windows(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    opened = {}

    def fake_startfile(path):
        opened["path"] = path

    monkeypatch.setattr(scanner.os, "startfile", fake_startfile, raising=False)
    monkeypatch.setattr(scanner.sys, "platform", "win32")
    scanner.open_pdf(Path("doc.pdf"))

    assert opened["path"] == "doc.pdf"

