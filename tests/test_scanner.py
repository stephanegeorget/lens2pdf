from types import SimpleNamespace
import importlib
import sys
from pathlib import Path
import numpy as np
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
    # time.  Provide a stub for ``cv2`` and ``pytesseract`` while using the real
    # NumPy implementation so image stacking logic can be exercised.
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "numpy", np)
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

    def fake_scan(
        *,
        gesture_enabled,
        boost_contrast,
        output_dir,
        timeout=None,
        stack_count=10,
    ):
        called["args"] = (
            gesture_enabled,
            boost_contrast,
            output_dir,
            stack_count,
        )

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(sys, "argv", ["scanner", "--no-gesture"])
    scanner.main()

    assert called["args"] == (False, True, None, 10)


def test_no_contrast_flag(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    called = {}

    def fake_scan(
        *,
        gesture_enabled,
        boost_contrast,
        output_dir,
        timeout=None,
        stack_count=10,
    ):
        called["args"] = (
            gesture_enabled,
            boost_contrast,
            output_dir,
            stack_count,
        )

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(sys, "argv", ["scanner", "--no-contrast"])
    scanner.main()

    assert called["args"] == (True, False, None, 10)


def test_output_dir_flag(monkeypatch, tmp_path):
    scanner = setup_fake_cv2(monkeypatch)
    called = {}

    def fake_scan(
        *,
        gesture_enabled,
        boost_contrast,
        output_dir,
        timeout=None,
        stack_count=10,
    ):
        called["args"] = (
            gesture_enabled,
            boost_contrast,
            output_dir,
            stack_count,
        )

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(
        sys,
        "argv",
        ["scanner", "--output-dir", str(tmp_path)],
    )
    scanner.main()

    assert called["args"] == (True, True, str(tmp_path), 10)


def test_default_timeout(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)
    called = {}

    def fake_scan(
        *,
        gesture_enabled,
        boost_contrast,
        output_dir,
        timeout=None,
        stack_count=10,
    ):
        called["timeout"] = timeout

    monkeypatch.setattr(scanner, "scan_document", fake_scan)
    monkeypatch.setattr(sys, "argv", ["scanner"])
    scanner.main()

    assert called["timeout"] == 60


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


def test_scan_document_reuses_camera(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)

    calls = {"list": 0, "select": 0, "open": 0}

    def fake_list():
        calls["list"] += 1
        return [(0, "cam")]

    def fake_select(_cams):
        calls["select"] += 1
        return 0

    class FakeCapture:
        def __init__(self, index):
            calls["open"] += 1

        def set(self, *_args):
            pass

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((1, 1, 3), dtype=np.uint8)

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(
        VideoCapture=FakeCapture,
        CAP_PROP_FRAME_WIDTH=0,
        CAP_PROP_FRAME_HEIGHT=0,
        imshow=lambda *a, **k: None,
        waitKey=lambda *a, **k: ord("s"),
        resize=lambda img, *a, **k: img,
        destroyAllWindows=lambda: None,
    )

    monkeypatch.setattr(scanner, "cv2", fake_cv2)
    monkeypatch.setattr(scanner, "list_cameras", fake_list)
    monkeypatch.setattr(scanner, "select_camera", fake_select)
    monkeypatch.setattr(scanner, "_create_window", lambda *_a: None)
    monkeypatch.setattr(scanner, "increase_contrast", lambda img: img)
    monkeypatch.setattr(scanner, "reduce_jpeg_artifacts", lambda img: img)
    monkeypatch.setattr(scanner, "save_pdf", lambda img, out: Path("out.pdf"))
    monkeypatch.setattr(scanner, "open_pdf", lambda _p: None)
    monkeypatch.setattr(scanner, "find_long_edges", lambda *a, **k: [])
    monkeypatch.setattr(scanner, "sys", SimpleNamespace(stdin=SimpleNamespace(read=lambda n: "")))
    monkeypatch.setattr(scanner, "PREVIEW_SCALE", 1.0)

    scanner.scan_document(gesture_enabled=False, boost_contrast=False)
    scanner.scan_document(gesture_enabled=False, boost_contrast=False)

    assert calls == {"list": 1, "select": 1, "open": 1}


def test_scan_document_stacks_frames(monkeypatch):
    scanner = setup_fake_cv2(monkeypatch)

    class FakeCapture:
        def __init__(self, index):
            self.count = 0

        def set(self, *_):
            pass

        def isOpened(self):
            return True

        def read(self):
            frame = np.full((1, 1, 3), self.count, dtype=np.uint8)
            self.count += 1
            return True, frame

        def release(self):
            pass

    fake_cv2 = SimpleNamespace(
        VideoCapture=FakeCapture,
        CAP_PROP_FRAME_WIDTH=0,
        CAP_PROP_FRAME_HEIGHT=0,
        imshow=lambda *a, **k: None,
        waitKey=lambda *a, **k: ord("s"),
        resize=lambda img, *a, **k: img,
        destroyAllWindows=lambda: None,
    )

    monkeypatch.setattr(scanner, "cv2", fake_cv2)
    monkeypatch.setattr(scanner, "list_cameras", lambda: [(0, "cam")])
    monkeypatch.setattr(scanner, "select_camera", lambda _c: 0)
    monkeypatch.setattr(scanner, "_create_window", lambda *_: None)
    monkeypatch.setattr(scanner, "find_long_edges", lambda *a, **k: [])
    monkeypatch.setattr(scanner, "increase_contrast", lambda img: img)
    monkeypatch.setattr(scanner, "reduce_jpeg_artifacts", lambda img: img)
    saved = {}

    def fake_save(img, out):
        saved["img"] = img.copy()
        return Path("out.pdf")

    monkeypatch.setattr(scanner, "save_pdf", fake_save)
    monkeypatch.setattr(scanner, "open_pdf", lambda _p: None)
    monkeypatch.setattr(scanner, "sys", SimpleNamespace(stdin=SimpleNamespace(read=lambda n: "")))
    monkeypatch.setattr(scanner, "PREVIEW_SCALE", 1.0)

    scanner.scan_document(
        gesture_enabled=False,
        boost_contrast=False,
        stack_count=3,
    )

    # Frames values were 0, 1 and 2 -> average = 1
    assert saved["img"][0, 0, 0] == 1

