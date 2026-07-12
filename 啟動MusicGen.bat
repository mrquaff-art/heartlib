@echo off
chcp 65001 >nul
title MusicGen Melody - Style Transfer
cd /d "%~dp0"

echo.
echo  ============================================
echo    MusicGen Melody - Style Transfer GUI
echo  ============================================
echo.

if not exist ".venv310\Scripts\python.exe" (
    echo  [ERROR] Python environment not found: .venv310
    pause
    exit /b 1
)

echo  [OK] Environment check passed.
echo  [OK] Starting GUI - browser will open automatically.
echo  [NOTE] First run will download model ~3.3GB from HuggingFace.
echo.

".venv310\Scripts\python.exe" musicgen_gui.py

echo.
echo  [GUI closed]
pause
