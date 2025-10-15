#!/bin/bash

echo "🎭 EmotionVision AI - Linux/Mac Launcher"
echo "======================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python detected"
echo

# Check if requirements are installed
python3 -c "import cv2, flask, tensorflow" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies ready"
echo

echo "🚀 Starting EmotionVision AI..."
echo "🌐 Open your browser and go to: http://localhost:5000"
echo "📱 Press Ctrl+C to stop the application"
echo

python3 app.py

