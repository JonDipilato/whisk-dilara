@echo off
REM ============================================================
REM   RUN GRANDMA STORY - Simple launcher
REM   Double-click to create a new episode!
REM ============================================================

REM Go to parent folder (main whisk folder)
cd /d "%~dp0.."

echo.
echo ===========================================================
echo   GRANDMA STORY VIDEO GENERATOR
echo ===========================================================
echo.

REM Activate virtual environment
if not exist "venv_win" (
    echo [ERROR] Setup not complete!
    echo Please run SETUP.bat first.
    echo.
    pause
    exit /b 1
)

call venv_win\Scripts\activate.bat

echo Starting the automation...
echo.
echo This will:
echo   1. Read your Excel file with scene prompts
echo   2. Generate character and environment images
echo   3. Generate all scene images (77 scenes - takes 2-3 hours)
echo   4. Create the final video
echo.
echo You can minimize this window but don't close it!
echo.

python excel_to_config.py --run

echo.
echo ===========================================================
echo   DONE! Check the output/episodes folder for your video.
echo ===========================================================
echo.
pause
