@echo off
REM ============================================================
REM   WHISK AUTOMATION - COMPLETE SETUP
REM   Double-click this file to install everything
REM ============================================================

setlocal enabledelayedexpansion

REM Go to parent folder (main whisk folder)
cd /d "%~dp0.."

echo.
echo ===========================================================
echo   WHISK AUTOMATION SETUP
echo   YouTube Story Video Generator
echo ===========================================================
echo.

REM ============================================================
REM STEP 1: Check Python
REM ============================================================
echo [Step 1/6] Checking Python installation...

where py >nul 2>nul
if errorlevel 1 (
    echo.
    echo ************************************************************
    echo   PYTHON NOT FOUND - PLEASE INSTALL IT FIRST
    echo ************************************************************
    echo.
    echo   1. Open your browser and go to:
    echo      https://www.python.org/downloads/
    echo.
    echo   2. Click the big yellow "Download Python" button
    echo.
    echo   3. Run the installer and IMPORTANT:
    echo      CHECK THE BOX "Add Python to PATH" at the bottom!
    echo.
    echo   4. Click "Install Now"
    echo.
    echo   5. When done, run this setup again
    echo.
    echo ************************************************************
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('py --version') do set PYTHON_VERSION=%%i
echo    OK - Python %PYTHON_VERSION% found

REM ============================================================
REM STEP 2: Check/Install FFmpeg
REM ============================================================
echo.
echo [Step 2/6] Checking FFmpeg (needed for video creation)...

where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo    FFmpeg not found. Attempting automatic install...

    where winget >nul 2>nul
    if errorlevel 1 (
        echo.
        echo ************************************************************
        echo   FFMPEG NOT INSTALLED - MANUAL INSTALL REQUIRED
        echo ************************************************************
        echo.
        echo   1. Open your browser and go to:
        echo      https://www.gyan.dev/ffmpeg/builds/
        echo.
        echo   2. Click "ffmpeg-release-essentials.zip" to download
        echo.
        echo   3. Extract the zip file to C:\ffmpeg
        echo.
        echo   4. Add FFmpeg to your PATH:
        echo      - Press Windows key, type "environment variables"
        echo      - Click "Edit the system environment variables"
        echo      - Click "Environment Variables" button
        echo      - Under "System variables", find "Path", click Edit
        echo      - Click "New" and add: C:\ffmpeg\bin
        echo      - Click OK on all windows
        echo.
        echo   5. RESTART YOUR COMPUTER
        echo.
        echo   6. Run this setup again
        echo.
        echo ************************************************************
        echo.
        pause
        exit /b 1
    )

    echo    Installing FFmpeg via Windows Package Manager...
    winget install Gyan.FFmpeg -e --silent --accept-package-agreements --accept-source-agreements

    if errorlevel 1 (
        echo.
        echo    [WARNING] Auto-install may have failed.
        echo    If video creation fails later, install FFmpeg manually.
    ) else (
        echo    OK - FFmpeg installed!
        echo.
        echo    NOTE: You may need to RESTART your computer for FFmpeg to work.
    )
) else (
    echo    OK - FFmpeg is already installed
)

REM ============================================================
REM STEP 3: Check Chrome
REM ============================================================
echo.
echo [Step 3/6] Checking Google Chrome...

set CHROME_FOUND=0
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set CHROME_FOUND=1
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set CHROME_FOUND=1
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set CHROME_FOUND=1

if %CHROME_FOUND%==0 (
    echo.
    echo ************************************************************
    echo   GOOGLE CHROME NOT FOUND - PLEASE INSTALL IT
    echo ************************************************************
    echo.
    echo   1. Go to: https://www.google.com/chrome/
    echo   2. Download and install Chrome
    echo   3. Run this setup again
    echo.
    echo ************************************************************
    echo.
    pause
    exit /b 1
)
echo    OK - Chrome found

REM ============================================================
REM STEP 4: Create Virtual Environment and Install Dependencies
REM ============================================================
echo.
echo [Step 4/6] Setting up Python packages...
echo    (This may take 2-5 minutes, please wait...)

if not exist "venv_win" (
    echo    Creating virtual environment...
    py -m venv venv_win
)

call venv_win\Scripts\activate.bat
python -m pip install --upgrade pip -q 2>nul

echo    Installing required packages...
pip install -q -r requirements.txt 2>nul
pip install -q selenium webdriver-manager 2>nul
pip install -q google-api-python-client google-auth-httplib2 google-auth-oauthlib 2>nul

echo    OK - All packages installed

REM ============================================================
REM STEP 5: Create Directory Structure
REM ============================================================
echo.
echo [Step 5/6] Creating project folders...

if not exist "data\environments" mkdir data\environments
if not exist "data\characters" mkdir data\characters
if not exist "output" mkdir output
if not exist "output\videos" mkdir output\videos
if not exist "output\audio" mkdir output\audio
if not exist "output\thumbnails" mkdir output\thumbnails
if not exist "output\episodes" mkdir output\episodes
if not exist "logs" mkdir logs
if not exist "assets\music" mkdir assets\music
if not exist "assets\music\calm" mkdir assets\music\calm

echo    OK - Folders created

REM ============================================================
REM STEP 6: Setup Chrome Profile for Whisk
REM ============================================================
echo.
echo [Step 6/6] Setting up Chrome for Google Whisk...

set CHROME_PROFILE=%LocalAppData%\Google\Chrome\User Data_Automation
if not exist "%CHROME_PROFILE%" mkdir "%CHROME_PROFILE%"

REM Update config.json with user's Chrome profile path
(
echo import json
echo config_path = 'config.json'
echo profile_path = r'%CHROME_PROFILE%'
echo with open(config_path, 'r'^) as f:
echo     config = json.load(f^)
echo config['browser']['user_data_dir'] = profile_path.replace('\\', '\\\\'^)
echo with open(config_path, 'w'^) as f:
echo     json.dump(config, f, indent=2^)
) > _temp_config_update.py
python _temp_config_update.py 2>nul
del _temp_config_update.py 2>nul

echo    OK - Chrome profile configured

REM ============================================================
REM DONE!
REM ============================================================
echo.
echo.
echo ===========================================================
echo   SETUP COMPLETE!
echo ===========================================================
echo.
echo.
echo ************************************************************
echo   FINAL STEP - LOG INTO GOOGLE
echo ************************************************************
echo.
echo   A Chrome window will open. You need to:
echo.
echo   1. Sign in with your Google account
echo   2. Go to the Whisk page and accept any terms
echo   3. Close Chrome when done
echo.
echo   Press any key to open Chrome...
echo.
pause

REM Find and launch Chrome
set CHROME_EXE=
if exist "%ProgramFiles%\Google\Chrome\Application\chrome.exe" set CHROME_EXE=%ProgramFiles%\Google\Chrome\Application\chrome.exe
if exist "%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe" set CHROME_EXE=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe
if exist "%LocalAppData%\Google\Chrome\Application\chrome.exe" set CHROME_EXE=%LocalAppData%\Google\Chrome\Application\chrome.exe

start "" "%CHROME_EXE%" --user-data-dir="%CHROME_PROFILE%" "https://labs.google/fx/tools/whisk"

echo.
echo ************************************************************
echo   Chrome is now open.
echo.
echo   1. Log into your Google account
echo   2. Accept any terms on the Whisk page
echo   3. Close Chrome
echo   4. Come back here and press any key
echo ************************************************************
echo.
pause

echo.
echo ===========================================================
echo   ALL DONE! YOU'RE READY TO CREATE VIDEOS!
echo ===========================================================
echo.
echo   To create a video, double-click:
echo      RUN_GRANDMA.bat
echo.
echo   Your videos will be saved in the "output\episodes" folder.
echo.
echo ===========================================================
echo.
pause
