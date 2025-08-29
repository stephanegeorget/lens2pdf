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


def test_reduce_jpeg_artifacts(monkeypatch):
    called = {}

    def fake_denoise(img, _dst, h, hColor, template, search):
        called["args"] = (h, hColor, template, search)
        return img

    fake_cv2 = types.SimpleNamespace(fastNlMeansDenoisingColored=fake_denoise)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    image_utils = importlib.import_module("src.image_utils")
    importlib.reload(image_utils)

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = image_utils.reduce_jpeg_artifacts(img)
    assert result is img
    assert called["args"] == (10, 10, 7, 21)


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


def test_find_document_contour_preview_draws_box():
    import cv2

    image_utils = importlib.import_module("src.image_utils")
    importlib.reload(image_utils)

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    rect = ((100, 100), (80, 60), 0)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(frame, [box], -1, (255, 255, 255), -1)

    preview = frame.copy()
    contour = image_utils.find_document_contour(
        frame, min_area_ratio=0.1, preview=preview
    )
    assert contour is not None
    assert np.any(np.all(preview == (0, 255, 0), axis=-1))
