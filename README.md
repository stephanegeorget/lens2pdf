# Cozy Python Starter

A minimal template for writing Python code using Visual Studio Code on a Windows machine.

## Features
- Pre-configured VS Code settings that point to a `.venv` interpreter.
- Example `src/main.py` module with a simple greeting function.
- Basic `pytest` unit test in `tests/`.

## Getting Started
1. Install [Python](https://www.python.org/downloads/windows/) 3.12 or newer and [Visual Studio Code](https://code.visualstudio.com/Download).
2. Clone this repository.
3. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. Open the folder in VS Code. The editor will automatically pick up the interpreter from `.venv`.
5. Use **Run > Start Debugging** or press `F5` to execute `src/main.py`. Run tests with **Terminal > Run Task > pytest** or `pytest` in the terminal.

## Running Tests
After the virtual environment is active:

```powershell
pytest
```

## Customization
Feel free to add linting, formatting, or additional libraries by editing `requirements.txt` and the VS Code configuration under `.vscode/`.

## Scanner

The repository includes an experimental document scanner in `src/scanner.py`.
To verify that your camera and GUI support are working, run the script with the
test flag:

```bash
python src/scanner.py --test-camera
```

This displays the raw camera feed. Press `q` to quit. Running the script
without the flag starts the full scanning workflow.
