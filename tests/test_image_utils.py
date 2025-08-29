import importlib
import types
import sys
import numpy as np


def test_increase_contrast(monkeypatch):
    def fake_convert(img, alpha, beta):
        return np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

    fake_cv2 = types.SimpleNamespace(convertScaleAbs=fake_convert)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    image_utils = importlib.import_module("src.image_utils")
    importlib.reload(image_utils)

    img = np.array([[100, 200], [50, 0]], dtype=np.uint8)
    result = image_utils.increase_contrast(img)
    expected = np.clip(img * 1.25, 0, 255).astype(np.uint8)
    assert np.array_equal(result, expected)


def test_find_document_contour_small_rotated():
    import cv2

    image_utils = importlib.import_module("src.image_utils")
    importlib.reload(image_utils)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    rect = ((100, 100), (80, 60), 30)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(frame, [box], -1, (255, 255, 255), -1)

    assert image_utils.find_document_contour(frame, min_area_ratio=0.5) is None

    contour = image_utils.find_document_contour(frame, min_area_ratio=0.1)
    assert contour is not None

    warped = image_utils.four_point_transform(frame, contour)
    h, w = warped.shape[:2]
    assert abs(w - 80) <= 5
    assert abs(h - 60) <= 5


def test_correct_orientation_fallback(monkeypatch):
    import cv2

    fake_pytesseract = types.SimpleNamespace(image_to_osd=lambda img: "Rotate: 0")
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

    image_utils = importlib.import_module("src.image_utils")
    importlib.reload(image_utils)
    monkeypatch.setattr(image_utils, "check_tesseract_installation", lambda: None)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.line(img, (50, 0), (50, 99), (255, 255, 255), 2)
    rotated = image_utils.rotate_bound(img, 30)

    corrected = image_utils.correct_orientation(rotated, None)
    angle = image_utils.detect_dominant_edge_angle(corrected)
    assert abs(angle) < 1
