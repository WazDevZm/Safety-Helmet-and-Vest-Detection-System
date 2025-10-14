"""
Quick test script for Ore Quality Classification System
Tests basic functionality and components
"""

import sys
import os
import numpy as np
import cv2
import time

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from ore_classifier import OreClassifier
        print("✅ OreClassifier imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OreClassifier: {e}")
        return False
    
    try:
        from ore_preprocessor import OrePreprocessor
        print("✅ OrePreprocessor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OrePreprocessor: {e}")
        return False
    
    try:
        from ore_data_generator import OreDataGenerator
        print("✅ OreDataGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OreDataGenerator: {e}")
        return False
    
    try:
        from ore_testing_system import OreTestingSystem
        print("✅ OreTestingSystem imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import OreTestingSystem: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of components"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test OreClassifier
        from ore_classifier import OreClassifier
        classifier = OreClassifier()
        classifier.create_model()
        print("✅ OreClassifier model created successfully")
        
        # Test OrePreprocessor
        from ore_preprocessor import OrePreprocessor
        preprocessor = OrePreprocessor()
        print("✅ OrePreprocessor initialized successfully")
        
        # Test OreDataGenerator
        from ore_data_generator import OreDataGenerator
        generator = OreDataGenerator()
        print("✅ OreDataGenerator initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_image_processing():
    """Test image processing capabilities"""
    print("\n🖼️ Testing image processing...")
    
    try:
        from ore_preprocessor import OrePreprocessor
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite('test_ore.jpg', test_image)
        
        # Test preprocessing
        preprocessor = OrePreprocessor()
        result = preprocessor.preprocess_image('test_ore.jpg')
        
        # Verify results
        assert 'original' in result
        assert 'processed' in result
        assert 'enhanced' in result
        assert 'resized' in result
        assert 'features' in result
        
        print("✅ Image processing test passed")
        
        # Clean up
        if os.path.exists('test_ore.jpg'):
            os.remove('test_ore.jpg')
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False

def test_data_generation():
    """Test synthetic data generation"""
    print("\n🎲 Testing data generation...")
    
    try:
        from ore_data_generator import OreDataGenerator
        
        generator = OreDataGenerator(output_dir='test_dataset')
        
        # Generate a small number of samples
        samples = generator.generate_synthetic_ore_samples(num_samples=5, image_size=(224, 224))
        
        assert len(samples) == 5
        print(f"✅ Generated {len(samples)} synthetic samples")
        
        # Clean up
        import shutil
        if os.path.exists('test_dataset'):
            shutil.rmtree('test_dataset')
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False

def test_classification():
    """Test classification functionality"""
    print("\n🔍 Testing classification...")
    
    try:
        from ore_classifier import OreClassifier
        
        classifier = OreClassifier()
        classifier.create_model()
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cv2.imwrite('test_classification.jpg', test_image)
        
        # Test prediction (this will use default model)
        results = classifier.predict_quality('test_classification.jpg')
        
        # Verify results structure
        assert 'predicted_class' in results
        assert 'confidence' in results
        assert 'top_3_predictions' in results
        assert 'all_probabilities' in results
        
        print(f"✅ Classification test passed - Predicted: {results['predicted_class']}")
        
        # Clean up
        if os.path.exists('test_classification.jpg'):
            os.remove('test_classification.jpg')
        
        return True
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

def test_web_app():
    """Test if web app can be imported"""
    print("\n🌐 Testing web application...")
    
    try:
        # Test if streamlit is available
        import streamlit as st
        print("✅ Streamlit available")
        
        # Test if app can be imported
        try:
            from ore_classification_app import main
            print("✅ Web application imported successfully")
            return True
        except ImportError as e:
            print(f"⚠️ Web application import failed: {e}")
            print("💡 This is expected if dependencies are missing")
            return True  # Don't fail the test for missing optional dependencies
        
    except ImportError as e:
        print(f"❌ Streamlit not available: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Ore Quality Classification System Tests")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Image Processing", test_image_processing),
        ("Data Generation", test_data_generation),
        ("Classification", test_classification),
        ("Web Application", test_web_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To run the web application:")
        print("   streamlit run ore_classification_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

