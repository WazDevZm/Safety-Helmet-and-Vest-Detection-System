"""
Test script for PPE detection system
Run this to test the detection functionality with sample images
"""

import cv2
import numpy as np
from ppe_detector import PPEDetector
import matplotlib.pyplot as plt

def create_sample_image():
    """Create a sample image for testing"""
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
    print("Testing PPE Detection System...")
    
    # Create detector
    detector = PPEDetector()
    
    # Create sample image
    sample_img = create_sample_image()
    
    print("Sample image created")
    
    # Run detection
    results = detector.detect_ppe(sample_img)
    
    print(f"Detection Results:")
    print(f"- Total persons detected: {results['total_persons']}")
    print(f"- Compliance rate: {results['ppe_compliance']['compliance_rate']:.1f}%")
    print(f"- Compliant persons: {results['ppe_compliance']['compliant_persons']}")
    print(f"- Missing helmets: {results['ppe_compliance']['missing_helmets']}")
    print(f"- Missing vests: {results['ppe_compliance']['missing_vests']}")
    
    # Draw results
    result_img = detector.draw_detections(sample_img, results['detections'])
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Detection Results")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_detection()
