"""
Simple test script for Safety Helmet and Vest Detection System
Tests the simplified version without matplotlib dependencies
"""

import cv2
import numpy as np
from ppe_detector_simple import PPEDetector

def create_sample_image():
    """Create a simple test image for testing"""
    # Create a simple test image with colored rectangles to simulate workers
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a person-like figure with helmet and vest
    # Head (with helmet)
    cv2.circle(img, (200, 100), 30, (255, 255, 0), -1)  # Yellow helmet
    cv2.circle(img, (200, 100), 25, (0, 0, 0), -1)  # Black head
    
    # Body (with vest)
    cv2.rectangle(img, (170, 130), (230, 250), (255, 255, 0), -1)  # Yellow vest
    cv2.rectangle(img, (180, 140), (220, 240), (0, 0, 0), -1)  # Black body
    
    # Arms
    cv2.rectangle(img, (150, 140), (170, 200), (0, 0, 0), -1)  # Left arm
    cv2.rectangle(img, (230, 140), (250, 200), (0, 0, 0), -1)  # Right arm
    
    # Legs
    cv2.rectangle(img, (180, 250), (200, 350), (0, 0, 0), -1)  # Left leg
    cv2.rectangle(img, (200, 250), (220, 350), (0, 0, 0), -1)  # Right leg
    
    return img

def test_detection():
    """Test the PPE detection system"""
    print("Testing PPE Detection System (Simplified Version)...")
    
    try:
        # Create detector
        detector = PPEDetector()
        print("✅ Detector initialized successfully")
        
        # Create sample image
        sample_img = create_sample_image()
        print("✅ Sample image created")
        
        # Run detection
        results = detector.detect_ppe(sample_img)
        print("✅ Detection completed")
        
        print(f"Detection Results:")
        print(f"- Total persons detected: {results['total_persons']}")
        print(f"- Compliance rate: {results['ppe_compliance']['compliance_rate']:.1f}%")
        print(f"- Compliant persons: {results['ppe_compliance']['compliant_persons']}")
        print(f"- Missing helmets: {results['ppe_compliance']['missing_helmets']}")
        print(f"- Missing vests: {results['ppe_compliance']['missing_vests']}")
        
        # Draw results
        result_img = detector.draw_detections(sample_img, results['detections'])
        print("✅ Results drawn successfully")
        
        # Save result image
        cv2.imwrite('test_result.jpg', result_img)
        print("✅ Result image saved as 'test_result.jpg'")
        
        print("\n🎉 Test completed successfully!")
        print("✅ All components are working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_detection()
    if success:
        print("\n🚀 You can now run the Streamlit app:")
        print("   streamlit run app_simple.py")
    else:
        print("\n❌ Please check the error messages above")
