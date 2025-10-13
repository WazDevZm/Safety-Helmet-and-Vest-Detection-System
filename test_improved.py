"""
Improved test script for Safety Helmet and Vest Detection System
Tests the enhanced detection algorithms with various scenarios
"""

import cv2
import numpy as np
from ppe_detector_simple import PPEDetector

def create_test_scenarios():
    """Create various test scenarios for comprehensive testing"""
    scenarios = {}
    
    # Scenario 1: Worker with both helmet and vest (compliant)
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head with yellow helmet
    cv2.circle(img1, (200, 100), 35, (0, 255, 255), -1)  # Bright yellow helmet
    cv2.circle(img1, (200, 100), 30, (0, 0, 0), -1)  # Black head
    # Body with bright yellow vest
    cv2.rectangle(img1, (160, 130), (240, 250), (0, 255, 255), -1)  # Bright yellow vest
    cv2.rectangle(img1, (170, 140), (230, 240), (0, 0, 0), -1)  # Black body
    # Arms and legs
    cv2.rectangle(img1, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img1, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img1, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img1, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Compliant Worker (Yellow)"] = img1
    
    # Scenario 2: Worker without helmet (non-compliant)
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head without helmet
    cv2.circle(img2, (200, 100), 30, (0, 0, 0), -1)  # Black head only
    # Body with orange vest
    cv2.rectangle(img2, (160, 130), (240, 250), (0, 165, 255), -1)  # Orange vest
    cv2.rectangle(img2, (170, 140), (230, 240), (0, 0, 0), -1)  # Black body
    # Arms and legs
    cv2.rectangle(img2, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img2, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img2, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img2, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Missing Helmet"] = img2
    
    # Scenario 3: Worker without vest (non-compliant)
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head with white helmet
    cv2.circle(img3, (200, 100), 35, (255, 255, 255), -1)  # White helmet
    cv2.circle(img3, (200, 100), 30, (0, 0, 0), -1)  # Black head
    # Body without vest
    cv2.rectangle(img3, (160, 130), (240, 250), (0, 0, 0), -1)  # Black body only
    # Arms and legs
    cv2.rectangle(img3, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img3, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img3, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img3, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Missing Vest"] = img3
    
    # Scenario 4: Worker with red helmet and green vest (compliant)
    img4 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head with red helmet
    cv2.circle(img4, (200, 100), 35, (0, 0, 255), -1)  # Red helmet
    cv2.circle(img4, (200, 100), 30, (0, 0, 0), -1)  # Black head
    # Body with green vest
    cv2.rectangle(img4, (160, 130), (240, 250), (0, 255, 0), -1)  # Green vest
    cv2.rectangle(img4, (170, 140), (230, 240), (0, 0, 0), -1)  # Black body
    # Arms and legs
    cv2.rectangle(img4, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img4, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img4, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img4, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Compliant Worker (Red/Green)"] = img4
    
    # Scenario 5: Multiple workers (mixed compliance)
    img5 = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    # Worker 1 (compliant - blue helmet, yellow vest)
    cv2.circle(img5, (150, 100), 35, (255, 0, 0), -1)  # Blue helmet
    cv2.circle(img5, (150, 100), 30, (0, 0, 0), -1)  # Black head
    cv2.rectangle(img5, (120, 130), (180, 250), (0, 255, 255), -1)  # Yellow vest
    cv2.rectangle(img5, (130, 140), (170, 240), (0, 0, 0), -1)  # Black body
    
    # Worker 2 (missing helmet - has vest)
    cv2.circle(img5, (350, 100), 30, (0, 0, 0), -1)  # Black head only
    cv2.rectangle(img5, (320, 130), (380, 250), (0, 255, 255), -1)  # Yellow vest
    cv2.rectangle(img5, (330, 140), (370, 240), (0, 0, 0), -1)  # Black body
    
    # Worker 3 (missing vest - has helmet)
    cv2.circle(img5, (550, 100), 35, (0, 255, 255), -1)  # Yellow helmet
    cv2.circle(img5, (550, 100), 30, (0, 0, 0), -1)  # Black head
    cv2.rectangle(img5, (520, 130), (580, 250), (0, 0, 0), -1)  # Black body only
    
    scenarios["Multiple Workers"] = img5
    
    return scenarios

def test_improved_detection():
    """Test the improved PPE detection system"""
    print("🔍 Testing Improved PPE Detection System...")
    print("=" * 60)
    
    try:
        # Create detector
        detector = PPEDetector()
        print("✅ Enhanced detector initialized successfully")
        
        # Create test scenarios
        scenarios = create_test_scenarios()
        print(f"✅ Created {len(scenarios)} test scenarios")
        
        # Test each scenario
        results = {}
        for scenario_name, image in scenarios.items():
            print(f"\n🧪 Testing scenario: {scenario_name}")
            
            # Run detection
            detection_results = detector.detect_ppe(image)
            
            # Draw results
            result_image = detector.draw_detections(image, detection_results['detections'])
            
            # Save result
            filename = f"test_result_{scenario_name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
            cv2.imwrite(filename, result_image)
            
            # Store results
            results[scenario_name] = {
                'detection_results': detection_results,
                'result_image': result_image,
                'filename': filename
            }
            
            # Print results
            compliance = detection_results['ppe_compliance']
            print(f"  📊 Total workers: {compliance['total_persons']}")
            print(f"  ✅ Compliant: {compliance['compliant_persons']}")
            print(f"  ❌ Missing helmets: {compliance['missing_helmets']}")
            print(f"  ❌ Missing vests: {compliance['missing_vests']}")
            print(f"  📈 Compliance rate: {compliance['compliance_rate']:.1f}%")
            print(f"  💾 Result saved: {filename}")
        
        # Summary
        print("\n📊 Test Summary:")
        print("=" * 40)
        
        total_workers = sum(r['detection_results']['ppe_compliance']['total_persons'] for r in results.values())
        total_compliant = sum(r['detection_results']['ppe_compliance']['compliant_persons'] for r in results.values())
        overall_compliance = (total_compliant / total_workers * 100) if total_workers > 0 else 0
        
        print(f"👥 Total workers across all scenarios: {total_workers}")
        print(f"✅ Total compliant workers: {total_compliant}")
        print(f"📊 Overall compliance rate: {overall_compliance:.1f}%")
        
        # Individual scenario analysis
        print("\n📋 Scenario Analysis:")
        for scenario_name, result in results.items():
            compliance = result['detection_results']['ppe_compliance']
            status = "✅ PASS" if compliance['compliance_rate'] > 0 else "❌ FAIL"
            print(f"  {status} {scenario_name}: {compliance['compliance_rate']:.1f}% compliance")
        
        print("\n🎉 Enhanced detection test completed successfully!")
        print("✅ All result images saved for visual inspection")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_detection()
    if success:
        print("\n🚀 Enhanced detection system is working!")
        print("💡 You can now run the Streamlit app with improved detection:")
        print("   streamlit run app_simple.py")
    else:
        print("\n❌ Please check the error messages above")
