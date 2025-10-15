"""
EmotionVision AI - Demo Runner
Quick demo without web interface
"""

import cv2
import numpy as np
import time
from emotion_detector import EmotionDetector

def run_demo():
    """Run the emotion detection demo"""
    print("🎭 EmotionVision AI - Demo Mode")
    print("=" * 40)
    print("📹 Starting camera...")
    print("🎯 Press 'q' to quit, 's' to save screenshot")
    print("=" * 40)
    
    # Initialize detector
    detector = EmotionDetector()
    detector.load_model()
    
    if not detector.start_camera():
        print("❌ Failed to start camera!")
        return
    
    screenshot_count = 0
    
    try:
        while True:
            frame, emotions = detector.get_frame()
            
            if frame is not None:
                # Add FPS counter
                fps = detector.camera.get(cv2.CAP_PROP_FPS) if detector.camera else 30
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add emotion info
                if emotions:
                    emotion = emotions[0]['emotion']
                    confidence = emotions[0]['confidence']
                    cv2.putText(frame, f"Emotion: {emotion}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Show frame
                cv2.imshow('EmotionVision AI - Demo', frame)
                
                # Print emotion info to console
                if emotions:
                    print(f"\r🎭 Detected: {emotions[0]['emotion']} ({emotions[0]['confidence']:.2f})", end="", flush=True)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"emotion_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n📸 Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()
        print("\n✅ Demo completed!")

if __name__ == "__main__":
    run_demo()

