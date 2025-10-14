"""
Setup script for Ore Quality Classification System
Helps users set up the system and verify installation
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported.")
        print("💡 Please install Python 3.8 or higher.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy',
        'opencv-python',
        'tensorflow',
        'streamlit',
        'matplotlib',
        'pandas',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("💡 Run: pip install -r requirements_ore.txt")
        return False
    
    print("✅ All required dependencies are available")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📥 Installing dependencies...")
    
    try:
        # Check if requirements file exists
        if not os.path.exists('requirements_ore.txt'):
            print("❌ requirements_ore.txt not found")
            return False
        
        # Install dependencies
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_ore.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'models',
        'data',
        'data/raw',
        'data/processed',
        'data/augmented',
        'results',
        'logs',
        'examples',
        'examples/sample_images',
        'examples/trained_models'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_system():
    """Test the system functionality"""
    print("\n🧪 Testing system functionality...")
    
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, 'test_ore_system.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System tests passed")
            return True
        else:
            print(f"❌ System tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def create_sample_data():
    """Create sample data for testing"""
    print("\n🎲 Creating sample data...")
    
    try:
        from ore_data_generator import OreDataGenerator
        
        generator = OreDataGenerator(output_dir='examples/sample_images')
        
        # Generate a small sample dataset
        samples = generator.generate_synthetic_ore_samples(
            num_samples=10, 
            image_size=(224, 224)
        )
        
        print(f"✅ Created {len(samples)} sample images")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. 🌐 Run the web application:")
    print("   streamlit run ore_classification_app.py")
    print("\n2. 📊 View sample data:")
    print("   Check the 'examples/sample_images' directory")
    print("\n3. 🧪 Run tests:")
    print("   python test_ore_system.py")
    print("\n4. 📚 Read documentation:")
    print("   Check README_ORE_CLASSIFICATION.md")
    
    print("\n🔗 Useful Commands:")
    print("   # Start web app")
    print("   streamlit run ore_classification_app.py")
    print("\n   # Run tests")
    print("   python test_ore_system.py")
    print("\n   # Generate sample data")
    print("   python -c \"from ore_data_generator import OreDataGenerator; OreDataGenerator().generate_synthetic_ore_samples(100)\"")

def main():
    """Main setup function"""
    print("🚀 Ore Quality Classification System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Installing missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return False
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        return False
    
    # Test system
    if not test_system():
        print("⚠️ System tests failed, but setup may still work")
    
    # Create sample data
    try:
        create_sample_data()
    except Exception as e:
        print(f"⚠️ Could not create sample data: {e}")
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

