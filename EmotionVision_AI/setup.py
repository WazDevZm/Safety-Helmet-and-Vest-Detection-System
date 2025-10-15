"""
EmotionVision AI - Setup Script
Automated installation and configuration
"""

import subprocess
import sys
import os
import platform

def install_requirements():
    """Install required packages"""
    print("🔧 Installing EmotionVision AI dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    print("📹 Checking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera detected and accessible!")
            cap.release()
            return True
        else:
            print("⚠️  Camera not accessible. Please check your camera connection.")
            return False
    except ImportError:
        print("❌ OpenCV not installed. Please run: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = ['models', 'data', 'logs', 'static/css', 'static/js', 'static/images']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ Created {directory}/")
    
    print("✅ All directories created!")

def main():
    """Main setup function"""
    print("🎭 EmotionVision AI - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if install_requirements():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Start emotion detection!")
        
        # Check camera
        check_camera()
        
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

