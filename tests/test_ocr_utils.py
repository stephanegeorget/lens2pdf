from types import SimpleNamespace
import importlib
import sys

import numpy as np


def test_save_pdf_uses_high_quality(monkeypatch, tmp_path):
    fake_cv2 = SimpleNamespace(cvtColor=lambda img, code: img, COLOR_BGR2RGB=0)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    import src.ocr_utils as ocr
    importlib.reload(ocr)

    called = {}

    def fake_image_to_pdf_or_hocr(img, extension, config):
        called["extension"] = extension
        called["config"] = config
        return b"%PDF-1.4"

    monkeypatch.setattr(
        ocr, "pytesseract", SimpleNamespace(image_to_pdf_or_hocr=fake_image_to_pdf_or_hocr)
    )
    monkeypatch.setattr(ocr, "check_tesseract_installation", lambda: None)
    monkeypatch.chdir(tmp_path)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    path = ocr.save_pdf(img)
    assert called["extension"] == "pdf"
    assert "--dpi 300" in called["config"]
    assert "jpg_quality=100" in called["config"]
    assert path.exists()


def test_save_pdf_custom_directory(monkeypatch, tmp_path):
    fake_cv2 = SimpleNamespace(cvtColor=lambda img, code: img, COLOR_BGR2RGB=0)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    import src.ocr_utils as ocr
    importlib.reload(ocr)

    called = {}

    def fake_image_to_pdf_or_hocr(img, extension, config):
        called["extension"] = extension
        called["config"] = config
        return b"%PDF-1.4"

    monkeypatch.setattr(
        ocr, "pytesseract", SimpleNamespace(image_to_pdf_or_hocr=fake_image_to_pdf_or_hocr)
    )
    monkeypatch.setattr(ocr, "check_tesseract_installation", lambda: None)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out_dir = tmp_path / "pdfs"
    path = ocr.save_pdf(img, out_dir)
    assert path.parent == out_dir
    assert path.exists()
