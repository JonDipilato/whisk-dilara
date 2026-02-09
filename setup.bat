@echo off
REM Whisk Automation - One-Command Setup for Windows
REM Double-click to run

setlocal enabledelayedexpansion

echo.
echo === Whisk Automation Setup ===
echo.

REM Check Python
where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9 or later from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('py --version') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%
echo.

REM Create virtual environment
if not exist "venv_win" (
    echo Creating virtual environment...
    py -m venv venv_win
) else (
    echo Virtual environment already exists.
)

REM Activate and install dependencies
echo Installing dependencies...
call venv_win\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -q -r requirements.txt

REM Create directories
echo Creating project structure...
if not exist "data\environments" mkdir data\environments
if not exist "data\characters" mkdir data\characters
if not exist "output" mkdir output
if not exist "logs" mkdir logs

REM Create .gitkeep files
type nul > data\environments\.gitkeep
type nul > data\characters\.gitkeep

REM Create default config if not exists
if not exist "config.json" (
    echo Creating default config.json...
    (
        echo {
        echo   "whisk_url": "https://labs.google/fx/tools/whisk/project",
        echo   "browser": {
        echo     "headless": false,
        echo     "slow_mo": 100,
        echo     "user_data_dir": null
        echo   },
        echo   "paths": {
        echo     "environments": "./data/environments",
        echo     "characters": "./data/characters",
        echo     "output": "./output",
        echo     "scenes_file": "./data/scenes.csv"
        echo   },
        echo   "generation": {
        echo     "images_per_prompt": 4,
        echo     "batches_per_scene": 2,
        echo     "image_format": "landscape",
        echo     "download_timeout": 60
        echo   },
        echo   "queue": {
        echo     "retry_on_failure": true,
        echo     "max_retries": 3,
        echo     "delay_between_scenes": 5
        echo     }
        echo }
    ) > config.json
)

REM Install Playwright browsers
echo Installing Playwright browsers...
playwright install chromium

echo.
echo === Setup Complete! ===
echo.
echo Next steps:
echo   1. Double-click run.bat to start the automation
echo   2. Or run commands:
echo      run.bat test
echo      run.bat --help
echo.
pause
