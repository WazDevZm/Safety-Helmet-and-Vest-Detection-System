"""
Debug script for PPE detection system
Helps understand what's happening in the detection process
"""

import cv2
import numpy as np
from ppe_detector_simple import PPEDetector

def create_realistic_test_image():
    """Create a more realistic test image that YOLOv8 can detect as a person"""
    # Create a larger, more realistic image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
    
    # Create a more human-like figure
    # Head (larger and more realistic)
    cv2.circle(img, (400, 150), 50, (220, 180, 140), -1)  # Skin color head
    cv2.circle(img, (400, 150), 45, (0, 0, 0), -1)  # Black head outline
    
    # Helmet (yellow hard hat)
    cv2.circle(img, (400, 150), 60, (0, 255, 255), -1)  # Yellow helmet
    cv2.circle(img, (400, 150), 55, (0, 0, 0), -1)  # Black helmet outline
    
    # Body (torso)
    cv2.rectangle(img, (350, 200), (450, 400), (0, 0, 0), -1)  # Black body
    
    # Safety vest (bright yellow)
    cv2.rectangle(img, (340, 210), (460, 390), (0, 255, 255), -1)  # Yellow vest
    cv2.rectangle(img, (350, 220), (450, 380), (0, 0, 0), -1)  # Black body under vest
    
    # Arms
    cv2.rectangle(img, (300, 220), (350, 350), (0, 0, 0), -1)  # Left arm
    cv2.rectangle(img, (450, 220), (500, 350), (0, 0, 0), -1)  # Right arm
    
    # Legs
    cv2.rectangle(img, (380, 400), (400, 550), (0, 0, 0), -1)  # Left leg
    cv2.rectangle(img, (400, 400), (420, 550), (0, 0, 0), -1)  # Right leg
    
    # Add some texture to make it more realistic
    # Add some noise to make it look more like a real image
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def debug_detection():
    """Debug the detection process step by step"""
    print("🔍 Debugging PPE Detection System...")
    print("=" * 50)
    
    try:
        # Create detector
        detector = PPEDetector()
        print("✅ Detector initialized")
        
        # Create test image
        test_img = create_realistic_test_image()
        print("✅ Test image created")
        
        # Save the test image
        cv2.imwrite('debug_test_image.jpg', test_img)
        print("💾 Test image saved as 'debug_test_image.jpg'")
        
        # Run YOLOv8 detection directly
        print("\n🔍 Running YOLOv8 person detection...")
        results = detector.model(test_img, verbose=True)
        
        print(f"📊 YOLOv8 detected {len(results)} result(s)")
        
        persons_detected = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                print(f"📦 Found {len(boxes)} bounding boxes")
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    print(f"  Box {i}: class={class_id}, confidence={confidence:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    
                    if class_id == 0 and confidence > 0.5:  # Person class
                        person_bbox = [int(x1), int(y1), int(x2), int(y2)]
                        persons_detected.append({
                            'bbox': person_bbox,
                            'confidence': confidence
                        })
        
        print(f"👥 Persons detected: {len(persons_detected)}")
        
        if persons_detected:
            print("\n🦺 Running PPE analysis...")
            # Run full PPE detection
            ppe_results = detector.detect_ppe(test_img)
            
            print(f"📊 PPE Analysis Results:")
            print(f"  Total persons: {ppe_results['total_persons']}")
            print(f"  Detections: {len(ppe_results['detections'])}")
            
            for i, detection in enumerate(ppe_results['detections']):
                print(f"  Person {i+1}:")
                print(f"    Helmet: {'✅' if detection['helmet_detected'] else '❌'}")
                print(f"    Vest: {'✅' if detection['vest_detected'] else '❌'}")
                print(f"    Compliant: {'✅' if detection['ppe_compliant'] else '❌'}")
                print(f"    Missing PPE: {detection['missing_ppe']}")
            
            # Draw results
            result_img = detector.draw_detections(test_img, ppe_results['detections'])
            cv2.imwrite('debug_result.jpg', result_img)
            print("💾 Result image saved as 'debug_result.jpg'")
        else:
            print("❌ No persons detected by YOLOv8")
            print("💡 This might be because the test image is too simple")
            print("💡 Try using a real photo of a person for testing")
        
        print("\n✅ Debug completed")
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_detection()
