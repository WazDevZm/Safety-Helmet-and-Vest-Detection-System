"""
Demo script for Safety Helmet and Vest Detection System
This script demonstrates the key features of the PPE detection system
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ppe_detector import PPEDetector
import time

def create_demo_scenarios():
    """Create different demo scenarios for testing"""
    scenarios = {}
    
    # Scenario 1: Worker with both helmet and vest (compliant)
    img1 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head with helmet
    cv2.circle(img1, (200, 100), 30, (255, 255, 0), -1)  # Yellow helmet
    cv2.circle(img1, (200, 100), 25, (0, 0, 0), -1)  # Black head
    # Body with vest
    cv2.rectangle(img1, (170, 130), (230, 250), (255, 255, 0), -1)  # Yellow vest
    cv2.rectangle(img1, (180, 140), (220, 240), (0, 0, 0), -1)  # Black body
    # Arms and legs
    cv2.rectangle(img1, (150, 140), (170, 200), (0, 0, 0), -1)
    cv2.rectangle(img1, (230, 140), (250, 200), (0, 0, 0), -1)
    cv2.rectangle(img1, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img1, (200, 250), (220, 350), (0, 0, 0), -1)
    
    scenarios["Compliant Worker"] = img1
    
    # Scenario 2: Worker without helmet (non-compliant)
    img2 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head without helmet
    cv2.circle(img2, (200, 100), 25, (0, 0, 0), -1)  # Black head only
    # Body with vest
    cv2.rectangle(img2, (170, 130), (230, 250), (255, 255, 0), -1)  # Yellow vest
    cv2.rectangle(img2, (180, 140), (220, 240), (0, 0, 0), -1)  # Black body
    # Arms and legs
    cv2.rectangle(img2, (150, 140), (170, 200), (0, 0, 0), -1)
    cv2.rectangle(img2, (230, 140), (250, 200), (0, 0, 0), -1)
    cv2.rectangle(img2, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img2, (200, 250), (220, 350), (0, 0, 0), -1)
    
    scenarios["Missing Helmet"] = img2
    
    # Scenario 3: Worker without vest (non-compliant)
    img3 = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Head with helmet
    cv2.circle(img3, (200, 100), 30, (255, 255, 0), -1)  # Yellow helmet
    cv2.circle(img3, (200, 100), 25, (0, 0, 0), -1)  # Black head
    # Body without vest
    cv2.rectangle(img3, (170, 130), (230, 250), (0, 0, 0), -1)  # Black body only
    # Arms and legs
    cv2.rectangle(img3, (150, 140), (170, 200), (0, 0, 0), -1)
    cv2.rectangle(img3, (230, 140), (250, 200), (0, 0, 0), -1)
    cv2.rectangle(img3, (180, 250), (200, 350), (0, 0, 0), -1)
    cv2.rectangle(img3, (200, 250), (220, 350), (0, 0, 0), -1)
    
    scenarios["Missing Vest"] = img3
    
    # Scenario 4: Multiple workers (mixed compliance)
    img4 = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    # Worker 1 (compliant)
    cv2.circle(img4, (150, 100), 30, (255, 255, 0), -1)  # Helmet
    cv2.circle(img4, (150, 100), 25, (0, 0, 0), -1)  # Head
    cv2.rectangle(img4, (120, 130), (180, 250), (255, 255, 0), -1)  # Vest
    cv2.rectangle(img4, (130, 140), (170, 240), (0, 0, 0), -1)  # Body
    
    # Worker 2 (missing helmet)
    cv2.circle(img4, (350, 100), 25, (0, 0, 0), -1)  # Head only
    cv2.rectangle(img4, (320, 130), (380, 250), (255, 255, 0), -1)  # Vest
    cv2.rectangle(img4, (330, 140), (370, 240), (0, 0, 0), -1)  # Body
    
    # Worker 3 (missing vest)
    cv2.circle(img4, (550, 100), 30, (255, 255, 0), -1)  # Helmet
    cv2.circle(img4, (550, 100), 25, (0, 0, 0), -1)  # Head
    cv2.rectangle(img4, (520, 130), (580, 250), (0, 0, 0), -1)  # Body only
    
    scenarios["Multiple Workers"] = img4
    
    return scenarios

def run_demo():
    """Run the complete demo"""
    print("🦺 Safety Helmet and Vest Detection System - Demo")
    print("=" * 60)
    
    # Initialize detector
    print("🔧 Initializing PPE detector...")
    detector = PPEDetector()
    print("✅ Detector initialized successfully")
    
    # Create demo scenarios
    print("\n📸 Creating demo scenarios...")
    scenarios = create_demo_scenarios()
    print(f"✅ Created {len(scenarios)} demo scenarios")
    
    # Process each scenario
    results = {}
    for scenario_name, image in scenarios.items():
        print(f"\n🔍 Processing scenario: {scenario_name}")
        
        start_time = time.time()
        detection_results = detector.detect_ppe(image)
        processing_time = time.time() - start_time
        
        # Draw detections
        result_image = detector.draw_detections(image, detection_results['detections'])
        
        results[scenario_name] = {
            'image': image,
            'result': result_image,
            'detection_results': detection_results,
            'processing_time': processing_time
        }
        
        # Print results
        compliance = detection_results['ppe_compliance']
        print(f"  📊 Total workers: {compliance['total_persons']}")
        print(f"  ✅ Compliant: {compliance['compliant_persons']}")
        print(f"  ❌ Missing helmets: {compliance['missing_helmets']}")
        print(f"  ❌ Missing vests: {compliance['missing_vests']}")
        print(f"  📈 Compliance rate: {compliance['compliance_rate']:.1f}%")
        print(f"  ⏱️  Processing time: {processing_time:.2f}s")
    
    # Display results
    print("\n📊 Displaying results...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (scenario_name, data) in enumerate(results.items()):
        # Original image
        axes[i].imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{scenario_name}\n(Original)")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Display detection results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (scenario_name, data) in enumerate(results.items()):
        # Detection results
        axes[i].imshow(cv2.cvtColor(data['result'], cv2.COLOR_BGR2RGB))
        compliance = data['detection_results']['ppe_compliance']
        title = f"{scenario_name}\nCompliance: {compliance['compliance_rate']:.1f}%"
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n📈 Demo Summary:")
    total_workers = sum(r['detection_results']['ppe_compliance']['total_persons'] for r in results.values())
    total_compliant = sum(r['detection_results']['ppe_compliance']['compliant_persons'] for r in results.values())
    overall_compliance = (total_compliant / total_workers * 100) if total_workers > 0 else 0
    
    print(f"  👥 Total workers across all scenarios: {total_workers}")
    print(f"  ✅ Total compliant workers: {total_compliant}")
    print(f"  📊 Overall compliance rate: {overall_compliance:.1f}%")
    
    avg_processing_time = np.mean([r['processing_time'] for r in results.values()])
    print(f"  ⏱️  Average processing time: {avg_processing_time:.2f}s")
    
    print("\n🎉 Demo completed successfully!")
    print("💡 To run the full web interface, use: streamlit run app.py")

if __name__ == "__main__":
    run_demo()
