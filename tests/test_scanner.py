from types import SimpleNamespace
import importlib
import sys


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
    registry = SimpleNamespace(getCameraInfoList=lambda: [info1, info2])
    fake_cv2 = SimpleNamespace(videoio_registry=registry, VideoCapture=FakeCapture)
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
    assert cams == [(0, "Generic Cam"), (1, "CZUR E012A")]
