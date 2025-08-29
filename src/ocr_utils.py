"""OCR helper utilities."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pytesseract
from PIL import Image

from types import SimpleNamespace


def check_tesseract_installation() -> None:
    """Ensure that the Tesseract executable is available."""
    cmd = shutil.which("tesseract")
    if cmd:
        return

    win_path = Path("C:/pf/Tesseract-OCR/tesseract.exe")
    if win_path.is_file():
        if not hasattr(pytesseract, "pytesseract"):
            pytesseract.pytesseract = SimpleNamespace()
        pytesseract.pytesseract.tesseract_cmd = str(win_path)
        return

    raise RuntimeError(
        "Tesseract OCR is required. Install it from "
        "https://github.com/UB-Mannheim/tesseract/wiki and ensure it "
        "is installed in C:\\pf\\Tesseract-OCR."
    )


def save_pdf(image) -> Path:
    """Save ``image`` with OCR text as a high-resolution PDF file."""
    check_tesseract_installation()
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img.info["dpi"] = (300, 300)
    pdf_bytes = pytesseract.image_to_pdf_or_hocr(pil_img, extension="pdf")
    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".pdf"
    path = Path(filename)
    path.write_bytes(pdf_bytes)
    return path


__all__ = ["check_tesseract_installation", "save_pdf"]
