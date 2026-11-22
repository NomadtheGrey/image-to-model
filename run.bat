@echo off
setlocal enabledelayedexpansion
title Image-to-3D Converter (TripoSR CPU)

echo ===================================================
echo     Local Image-to-3D Model Converter (CPU)
echo ===================================================
echo.

cd /d "%~dp0"

:: 1. Locate Python executable
set "PYTHON_EXE="

:: Check user-installed Python 3.11 first
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
)

:: If not found, check system PATH
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python was not found on your system!
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Using Python: !PYTHON_EXE!
!PYTHON_EXE! --version

:: 2. Check if dependencies are installed (check gradio and torch)
!PYTHON_EXE! -c "import gradio, torch, rembg, trimesh, tsr" >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [SETUP] Installing required dependencies from requirements.txt...
    echo (This may take a few minutes on initial setup)
    echo.
    !PYTHON_EXE! -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies. Check output above.
        pause
        exit /b 1
    )
    echo [SETUP] Dependencies installed successfully.
    echo.
)

:: 3. Launch application
echo.
echo [LAUNCH] Starting local Gradio web app on http://127.0.0.1:7860...
echo Keep this window open while using the application.
echo.

!PYTHON_EXE! app.py

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Application exited with code !errorlevel!.
    pause
)
