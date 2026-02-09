============================================================
   WHISK AUTOMATION - SETUP GUIDE
   YouTube Story Video Generator
============================================================

Hi! This will help you create animated story videos
automatically using AI image generation.


WHAT YOU NEED FIRST
-------------------
1. Windows 10 or 11
2. Google Chrome browser
3. A Google account (for Google Whisk AI)
4. Internet connection


HOW TO SET UP (ONE TIME ONLY)
-----------------------------
1. Double-click SETUP.bat

2. Follow the on-screen instructions:
   - It will check if Python is installed (if not, it tells you how)
   - It will check if FFmpeg is installed (if not, it tries to install it)
   - It will install all the required software
   - At the end, Chrome will open for you to log in

3. When Chrome opens:
   - Sign in with your Google account
   - The Whisk page will load - accept any terms if asked
   - Close Chrome when done

4. That's it! Setup is complete.


HOW TO CREATE A VIDEO
---------------------
1. Double-click RUN_GRANDMA.bat

2. The automation will:
   - Read your story prompts from the Excel file
   - Generate character images
   - Generate all scene images (this takes 2-3 hours!)
   - Create the final video

3. You can minimize the window but DON'T close it!

4. When done, find your video in:
   output/episodes/grandma_epX_YYYYMMDD_HHMMSS/


TROUBLESHOOTING
---------------
"Python not found"
   -> Install Python from python.org
   -> Make sure to check "Add to PATH" during install

"FFmpeg not found"
   -> Download from gyan.dev/ffmpeg/builds
   -> Extract to C:\ffmpeg
   -> Add C:\ffmpeg\bin to your PATH
   -> Restart computer

"Chrome not found"
   -> Install Google Chrome from google.com/chrome

Video creation fails
   -> Make sure you're logged into Google in the automation Chrome
   -> Try running SETUP.bat again to re-login


QUESTIONS?
----------
Ask Jon - he set this up for you!


============================================================
