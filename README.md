# lens2pdf

lens2pdf turns an overhead webcam into a document scanner. Point the camera at a
page and show a **V sign** with your hand (or press `s`) to capture. The program
detects the page edges and orientation, boosts contrast, performs OCR with
Tesseract and saves a searchable PDF. Perspective is not corrected, so keep the
camera parallel to the document for best results.

## Features

- **Gesture triggered scans** – hold up a V sign to start a 3‑2‑1 countdown.
- **Automatic document detection** – edges are highlighted so you can align the
  page.
- **Orientation and OCR** – uses [Tesseract](https://github.com/tesseract-ocr/tesseract)
  to rotate pages upright and embed recognised text.
- **Contrast boost and frame stacking** – improves legibility and reduces noise.
- **Opens PDFs automatically** – each capture is written to a timestamped PDF.
- **Overhead camera recommended** – works best with devices like the CZUR Lens
  and other cameras mounted perpendicular to the page.

## Requirements

- Python 3.12 or newer
- Tesseract OCR
- FFmpeg (for camera discovery)
- Python packages listed in `requirements.txt`

## Quick start

```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m src.scanner --help
```

The `--help` output documents all options including camera testing, disabling
gestures, edge detection thresholds and frame stacking.

Run the scanner:

```bash
python -m src.scanner
```

Hold a document in view and show a V sign (or press `s`) to capture. Press `q`
to quit.

### Preview window and performance

The preview window is scaled to 25% of the camera resolution and the capture
defaults to 1600×1200 to keep the video feed responsive. These values can be
adjusted in `src/scanner.py` via `PREVIEW_SCALE`, `CAPTURE_WIDTH` and
`CAPTURE_HEIGHT` if your hardware performs better at different settings.

## Tests

```bash
pytest
```

