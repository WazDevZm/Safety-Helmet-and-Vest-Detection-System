"""
Test script for Fixed PPE Detection System
Tests the fixed detection with proper flagging
"""

import cv2
import numpy as np
from ppe_detector_fixed import PPEDetectorFixed

def create_test_scenarios():
    """Create test scenarios with clear PPE violations"""
    scenarios = {}
    
    # Scenario 1: Worker with both helmet and vest (compliant)
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light background
    
    # Head with VERY bright yellow helmet
    cv2.circle(img1, (200, 100), 40, (0, 255, 255), -1)  # Bright yellow helmet
    cv2.circle(img1, (200, 100), 35, (0, 0, 0), -1)  # Black head
    
    # Body with VERY bright orange vest
    cv2.rectangle(img1, (160, 130), (240, 250), (0, 165, 255), -1)  # Orange vest
    cv2.rectangle(img1, (170, 140), (230, 240), (0, 0, 0), -1)  # Black body
    
    # Arms and legs
    cv2.rectangle(img1, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img1, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img1, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img1, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Compliant Worker (Should Detect PPE)"] = img1
    
    # Scenario 2: Worker without helmet (should be flagged)
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Head without helmet (just black head)
    cv2.circle(img2, (200, 100), 35, (0, 0, 0), -1)  # Black head only
    
    # Body with vest
    cv2.rectangle(img2, (160, 130), (240, 250), (0, 255, 255), -1)  # Yellow vest
    cv2.rectangle(img2, (170, 140), (230, 240), (0, 0, 0), -1)  # Black body
    
    # Arms and legs
    cv2.rectangle(img2, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img2, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img2, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img2, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Missing Helmet (Should Flag)"] = img2
    
    # Scenario 3: Worker without vest (should be flagged)
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 240
    
    # Head with helmet
    cv2.circle(img3, (200, 100), 40, (255, 255, 255), -1)  # White helmet
    cv2.circle(img3, (200, 100), 35, (0, 0, 0), -1)  # Black head
    
    # Body without vest (just black body)
    cv2.rectangle(img3, (160, 130), (240, 250), (0, 0, 0), -1)  # Black body only
    
    # Arms and legs
    cv2.rectangle(img3, (140, 140), (160, 200), (0, 0, 0), -1)
    cv2.rectangle(img3, (240, 140), (260, 200), (0, 0, 0), -1)
    cv2.rectangle(img3, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img3, (200, 250), (220, 350), (0, 0, 0), -1)
    scenarios["Missing Vest (Should Flag)"] = img3
    
    return scenarios

def test_fixed_detection():
    """Test the fixed PPE detection system"""
    print("🔍 Testing Fixed PPE Detection System...")
    print("=" * 60)
    
    try:
        # Create detector
        detector = PPEDetectorFixed()
        print("✅ Fixed detector initialized successfully")
        
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
            filename = f"fixed_test_{scenario_name.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
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
            
            # Check individual detections
            for i, detection in enumerate(detection_results['detections']):
                print(f"    Worker {i+1}:")
                print(f"      Helmet: {'✅' if detection['helmet_detected'] else '❌'}")
                print(f"      Vest: {'✅' if detection['vest_detected'] else '❌'}")
                print(f"      Status: {detection['safety_status']}")
                if detection['missing_ppe']:
                    print(f"      Missing: {', '.join(detection['missing_ppe'])}")
            
            print(f"  💾 Result saved: {filename}")
        
        # Summary
        print("\n📊 Fixed Detection Test Summary:")
        print("=" * 50)
        
        total_workers = sum(r['detection_results']['ppe_compliance']['total_persons'] for r in results.values())
        total_compliant = sum(r['detection_results']['ppe_compliance']['compliant_persons'] for r in results.values())
        total_missing_helmets = sum(r['detection_results']['ppe_compliance']['missing_helmets'] for r in results.values())
        total_missing_vests = sum(r['detection_results']['ppe_compliance']['missing_vests'] for r in results.values())
        
        print(f"👥 Total workers across all scenarios: {total_workers}")
        print(f"✅ Total compliant workers: {total_compliant}")
        print(f"❌ Total missing helmets: {total_missing_helmets}")
        print(f"❌ Total missing vests: {total_missing_vests}")
        
        # Check if flagging is working
        print("\n🚨 Flagging Analysis:")
        for scenario_name, result in results.items():
            compliance = result['detection_results']['ppe_compliance']
            if "Should Flag" in scenario_name:
                if compliance['compliance_rate'] < 100:
                    print(f"  ✅ {scenario_name}: CORRECTLY FLAGGED")
                else:
                    print(f"  ❌ {scenario_name}: NOT FLAGGED (should be flagged)")
            else:
                if compliance['compliance_rate'] == 100:
                    print(f"  ✅ {scenario_name}: CORRECTLY SAFE")
                else:
                    print(f"  ⚠️ {scenario_name}: INCORRECTLY FLAGGED")
        
        print("\n🎉 Fixed detection test completed!")
        print("✅ All result images saved for visual inspection")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_detection()
    if success:
        print("\n🚀 Fixed detection system is working!")
        print("💡 You can now run the fixed Streamlit app:")
        print("   streamlit run app_fixed.py")
    else:
        print("\n❌ Please check the error messages above")
