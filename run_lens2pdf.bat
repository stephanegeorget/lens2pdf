@echo off
REM ==============================================================
REM  lens2pdf launcher
REM  This batch file activates the local virtual environment and
REM  starts the Python entry point for the project.
REM
REM  Customize the settings below to match your setup.
REM ==============================================================

REM Change directory to the location of this script
cd /d "%~dp0"

REM Path to the virtual environment's "Scripts" folder
set "VENV_SCRIPTS=.venv\Scripts"

REM Activate the virtual environment if it exists
if exist "%VENV_SCRIPTS%\activate.bat" (
    call "%VENV_SCRIPTS%\activate.bat"
) else (
    echo [INFO] No virtual environment found at %VENV_SCRIPTS%. Using system Python.
)

REM Launch the main application. Replace the script or add arguments as needed.
python src\main.py %*

