@echo off
chcp 65001 >nul
title HeartMuLa Music Generator
cd /d "%~dp0"

echo.
echo  ============================================
echo    HeartMuLa Music Generator - Starting...
echo  ============================================
echo.

if not exist ".venv310\Scripts\python.exe" (
    echo  [ERROR] Python environment not found: .venv310
    echo  Please make sure you are in the heartlib directory.
    pause
    exit /b 1
)

if not exist "ckpt\gen_config.json" (
    echo  [ERROR] Model files not found in ckpt folder.
    echo  Please download the checkpoints first.
    pause
    exit /b 1
)

echo  [OK] Environment check passed.
echo  [OK] Starting GUI - browser will open at http://127.0.0.1:7860
echo.

".venv310\Scripts\python.exe" app_gui.py

echo.
echo  [GUI closed]
pause
