# lens2pdf

lens2pdf turns any webcam into a document scanner. Point the camera at a page and show a **V sign** with your hand to capture the image. The app detects the document edges, corrects perspective and orientation, enhances contrast, performs OCR with Tesseract and saves a searchable PDF.

## Features
- **Gesture triggered scans** – hold up a V (peace) sign to start a 3‑2‑1 countdown and capture a page. You can also press `s` on the keyboard.
- **Automatic document detection** – locates paper within the frame and performs a perspective transform to flatten it.
- **Orientation and OCR** – uses [Tesseract](https://github.com/tesseract-ocr/tesseract) to rotate pages upright and embed recognized text for searching.
- **Contrast boost** – improves legibility of lighter documents.
- **Opens the PDF automatically** – each capture is written to a timestamped PDF and opened with the system viewer.

## Requirements
### Software
- A webcam.
- [Python](https://www.python.org/downloads/) **3.12 or newer**.
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). On first run the program checks for the `tesseract` executable and shows an error if it is missing.
- On Linux you may also need system packages such as `libgl1` for OpenCV.

### Python packages
The project relies on the packages listed in `requirements.txt`:

```
opencv-python
numpy
mediapipe
pytesseract
Pillow
pytest  # tests only
```

These will be installed in the installation step below.

## Installation
The steps below assume no prior Python knowledge.

1. **Install Python**
   - Download Python from [python.org](https://www.python.org/downloads/).
   - **Windows**: during setup check *Add Python to PATH*. After installation open *Command Prompt* and run `python --version` to verify.

2. **Install Tesseract OCR**
   - **Windows**: use the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) and accept the default path `C:\\pf\\Tesseract-OCR`.
   - **macOS**: `brew install tesseract`.
   - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`.
   - Confirm with `tesseract --version` in a terminal.

3. **Get the lens2pdf code**
   - EITHER click the **Code ▾** button on GitHub and download the ZIP, then unzip it.
   - OR install [Git](https://git-scm.com/downloads) and run:
     ```
     git clone <repository-url>
     cd lens2pdf
     ```

4. **Create a virtual environment** (isolated place to install Python packages)
   - **Windows**:
     ```
     python -m venv .venv
     .\venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```

5. **Install the Python dependencies**
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running
- Start the scanner:
  - **Windows**: double‑click `run_lens2pdf.bat` or run `python src/scanner.py` from an activated virtual environment.
  - **macOS/Linux**: run `python src/scanner.py` from an activated virtual environment.
- Optional camera test: `python src/scanner.py --test-camera`.
- Position your document in view. When ready, show a **V sign** (or press `s`). The app captures the page, saves a PDF in the current directory and opens it. Press `q` to quit.

## Running tests
After activating the virtual environment you can run:
```
pytest
```

## License
This project is provided as-is without any warranty.
