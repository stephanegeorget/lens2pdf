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
