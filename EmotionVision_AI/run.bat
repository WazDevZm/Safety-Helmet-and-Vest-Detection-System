@echo off
echo 🎭 EmotionVision AI - Windows Launcher
echo =====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected
echo.

REM Check if requirements are installed
python -c "import cv2, flask, tensorflow" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✅ Dependencies ready
echo.

echo 🚀 Starting EmotionVision AI...
echo 🌐 Open your browser and go to: http://localhost:5000
echo 📱 Press Ctrl+C to stop the application
echo.

python app.py

pause

