"""
Installation script for Safety Helmet and Vest Detection System
Automated setup and dependency installation
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"💾 Available RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM available. Performance may be affected.")
        else:
            print("✅ Sufficient RAM available")
    except ImportError:
        print("⚠️  psutil not available for memory check")
    
    # Check platform
    system = platform.system()
    print(f"🖥️  Operating System: {system}")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        print("⬆️  Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("📥 Installing project dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("💡 Try running: pip install -r requirements.txt manually")
        return False

def verify_installation():
    """Verify that all components are working"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test imports
        print("  Testing OpenCV...")
        import cv2
        print(f"    ✅ OpenCV version: {cv2.__version__}")
        
        print("  Testing YOLOv8...")
        from ultralytics import YOLO
        print("    ✅ YOLOv8 imported successfully")
        
        print("  Testing Streamlit...")
        import streamlit
        print(f"    ✅ Streamlit version: {streamlit.__version__}")
        
        print("  Testing NumPy...")
        import numpy as np
        print(f"    ✅ NumPy version: {np.__version__}")
        
        print("  Testing Matplotlib...")
        import matplotlib
        print(f"    ✅ Matplotlib version: {matplotlib.__version__}")
        
        print("  Testing Seaborn...")
        import seaborn
        print(f"    ✅ Seaborn version: {seaborn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "uploads",
        "outputs", 
        "temp",
        "logs",
        "models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created directory: {directory}")

def run_tests():
    """Run basic tests to verify functionality"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test PPE detector
        print("  Testing PPE detector...")
        from ppe_detector import PPEDetector
        detector = PPEDetector()
        print("    ✅ PPE detector initialized")
        
        # Test with sample image
        print("  Testing with sample image...")
        import numpy as np
        sample_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        results = detector.detect_ppe(sample_img)
        print("    ✅ Detection test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main installation function"""
    print("🦺 Safety Helmet and Vest Detection System - Installation")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please upgrade Python to 3.8+")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run tests
    if not run_tests():
        print("⚠️  Basic tests failed, but installation may still work")
    
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next steps:")
    print("  1. Run the application: streamlit run app.py")
    print("  2. Or use the run script: python run.py")
    print("  3. Open your browser to http://localhost:8501")
    print("  4. Test with sample images")
    
    print("\n🔧 Available commands:")
    print("  • streamlit run app.py          - Start web application")
    print("  • python run.py                 - Easy run script")
    print("  • python test_detection.py       - Run basic tests")
    print("  • python demo.py                - Run comprehensive demo")
    print("  • python benchmark.py            - Performance benchmark")

if __name__ == "__main__":
    main()
