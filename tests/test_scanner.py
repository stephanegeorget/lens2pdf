from types import SimpleNamespace
import importlib
import sys
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

