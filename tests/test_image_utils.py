import importlib
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


def test_find_long_edges_detects_axes(monkeypatch):
    def fake_cvtColor(img, code):
        return img[:, :, 0]

    def fake_blur(img, k, s):
        return img

    def fake_canny(img, t1, t2):
        return img

    def fake_hough(edges, rho, theta, **kwargs):
        return np.array([[[0, 100, 199, 100]], [[100, 0, 100, 199]]])

    fake_cv2 = types.SimpleNamespace(
        cvtColor=fake_cvtColor,
        COLOR_BGR2GRAY=0,
        GaussianBlur=fake_blur,
        Canny=fake_canny,
        HoughLinesP=fake_hough,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    image_utils = importlib.import_module("src.image_utils")
    importlib.reload(image_utils)

    frame = np.full((200, 200, 3), 255, dtype=np.uint8)

    edges = image_utils.find_long_edges(frame, min_length_ratio=0.2)
    angles = [angle for *_coords, angle in edges]
    assert any(abs(a) <= 3 for a in angles)
    assert any(abs(a - 90) <= 3 for a in angles)
