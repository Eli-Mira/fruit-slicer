@echo off
rem Runs the Gesture Slicer game using the first Python that has pygame.
rem Falls back to a clear install hint instead of a confusing import error.

setlocal
cd /d "%~dp0"

rem 1) Explicit venv override via env var GAME_PYTHON (full path to python.exe).
if defined GAME_PYTHON (
    "%GAME_PYTHON%" -c "import pygame" >nul 2>&1
    if errorlevel 1 (
        echo [run_game] GAME_PYTHON interpreter has no pygame: %GAME_PYTHON%
        goto :hint
    )
    "%GAME_PYTHON%" game.py %*
    goto :eof
)

rem 2) Known gesture-control venv (has pygame + cv2 + mediapipe).
set "KNOWN_VENV=%USERPROFILE%\gesture_control_app\.venv\Scripts\python.exe"
if exist "%KNOWN_VENV%" (
    "%KNOWN_VENV%" -c "import pygame" >nul 2>&1
    if not errorlevel 1 (
        echo [run_game] Using %KNOWN_VENV%
        "%KNOWN_VENV%" game.py %*
        goto :eof
    )
)

rem 3) Project-local venv.
set "LOCAL_VENV=%~dp0..\.venv\Scripts\python.exe"
if exist "%LOCAL_VENV%" (
    "%LOCAL_VENV%" -c "import pygame" >nul 2>&1
    if not errorlevel 1 (
        "%LOCAL_VENV%" game.py %*
        goto :eof
    )
)

rem 4) System python.
python -c "import pygame" >nul 2>&1
if not errorlevel 1 (
    python game.py %*
    goto :eof
)

:hint
echo.
echo [run_game] ERROR: pygame is not installed in any of the searched interpreters.
echo.
echo   Fix: install the game dependencies into your Python environment:
echo     python -m pip install -r requirements.txt
echo.
echo   (Or point this script at a venv that already has pygame:)
echo     run_game.bat "C:\path\to\venv\Scripts\python.exe"
echo.
exit /b 1