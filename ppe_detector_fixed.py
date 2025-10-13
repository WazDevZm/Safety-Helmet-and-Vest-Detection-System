"""
Fixed Safety Helmet and Vest Detection System
Properly flags missing PPE with strict detection criteria
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Any
import os

class PPEDetectorFixed:
    """
    Fixed PPE Detection System that properly flags missing safety equipment
    """
    
    def __init__(self, model_path: str = None):
        """Initialize the fixed PPE detector"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLOv8 model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
        
        # Colors for bounding boxes
        self.colors = {
            'helmet': (0, 255, 0),      # Green for helmet
            'vest': (255, 0, 0),        # Blue for vest
            'missing_helmet': (0, 0, 255),  # Red for missing helmet
            'missing_vest': (0, 165, 255),   # Orange for missing vest
            'person': (255, 255, 255),  # White for person
            'unsafe': (0, 0, 255),      # Red for unsafe
            'safe': (0, 255, 0)         # Green for safe
        }
    
    def detect_ppe(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Fixed PPE detection that properly flags missing equipment
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        try:
            # Run YOLOv8 inference
            results = self.model(image, verbose=False)
            
            # Process results
            detections = []
            persons_detected = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only process person detections
                        if class_id == 0 and confidence > 0.3:
                            person_bbox = [int(x1), int(y1), int(x2), int(y2)]
                            persons_detected.append({
                                'bbox': person_bbox,
                                'confidence': confidence
                            })
            
            # If no persons detected, create a default person region for testing
            if not persons_detected:
                h, w = image.shape[:2]
                # Create a person region in the center
                center_x, center_y = w // 2, h // 2
                person_w, person_h = w // 3, h // 2
                x1 = max(0, center_x - person_w // 2)
                y1 = max(0, center_y - person_h // 2)
                x2 = min(w, center_x + person_w // 2)
                y2 = min(h, center_y + person_h // 2)
                
                persons_detected.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': 0.5
                })
                print("⚠️ No persons detected by YOLOv8, using default region")
            
            # Analyze each detected person for PPE with STRICT criteria
            for person in persons_detected:
                person_analysis = self._analyze_person_ppe_strict(image, person)
                detections.append(person_analysis)
            
            return {
                'detections': detections,
                'total_persons': len(persons_detected),
                'ppe_compliance': self._calculate_compliance(detections)
            }
            
        except Exception as e:
            print(f"Error in PPE detection: {e}")
            return {
                'detections': [],
                'total_persons': 0,
                'ppe_compliance': {
                    'compliance_rate': 0.0,
                    'total_persons': 0,
                    'compliant_persons': 0,
                    'missing_helmets': 0,
                    'missing_vests': 0
                }
            }
    
    def _analyze_person_ppe_strict(self, image: np.ndarray, person: Dict) -> Dict:
        """
        STRICT person PPE analysis that properly flags missing equipment
        
        Args:
            image: Input image
            person: Person detection data
            
        Returns:
            Person analysis with PPE status
        """
        x1, y1, x2, y2 = person['bbox']
        
        # Extract person region with padding
        padding = 10
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(image.shape[1], x2 + padding)
        y2_padded = min(image.shape[0], y2 + padding)
        
        person_region = image[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Analyze for helmet (head region) - STRICT
        head_height = int(person_region.shape[0] * 0.35)
        helmet_region = person_region[:head_height, :]
        helmet_detected = self._detect_helmet_strict(helmet_region)
        
        # Analyze for vest (torso region) - STRICT
        vest_start = int(person_region.shape[0] * 0.15)
        vest_end = int(person_region.shape[0] * 0.85)
        vest_region = person_region[vest_start:vest_end, :]
        vest_detected = self._detect_vest_strict(vest_region)
        
        # Size validation - must be large enough
        min_region_size = 30
        if helmet_region.shape[0] < min_region_size or helmet_region.shape[1] < min_region_size:
            helmet_detected = False
        if vest_region.shape[0] < min_region_size or vest_region.shape[1] < min_region_size:
            vest_detected = False
        
        return {
            'person_bbox': person['bbox'],
            'confidence': person['confidence'],
            'helmet_detected': helmet_detected,
            'vest_detected': vest_detected,
            'ppe_compliant': helmet_detected and vest_detected,
            'missing_ppe': self._get_missing_ppe(helmet_detected, vest_detected),
            'safety_status': 'SAFE' if (helmet_detected and vest_detected) else 'UNSAFE'
        }
    
    def _detect_helmet_strict(self, head_region: np.ndarray) -> bool:
        """
        STRICT helmet detection - only detects obvious bright helmets
        
        Args:
            head_region: Image region containing the head
            
        Returns:
            True if helmet is detected with high confidence
        """
        if head_region.size == 0:
            return False
        
        try:
            # Method 1: STRICT color detection - only very bright colors
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Only very bright, obvious helmet colors
            helmet_colors = [
                ([20, 200, 200], [30, 255, 255]),  # Very bright yellow
                ([0, 0, 220], [180, 30, 255]),     # Very bright white
                ([0, 200, 200], [10, 255, 255]),   # Very bright red
                ([100, 200, 200], [130, 255, 255]), # Very bright blue
                ([40, 200, 200], [80, 255, 255]),  # Very bright green
                ([10, 220, 220], [25, 255, 255])   # Very bright orange
            ]
            
            color_score = 0
            for lower, upper in helmet_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = np.sum(mask)
                if pixel_count > 2000:  # High threshold
                    color_score += pixel_count
            
            # Must have very significant bright color presence
            if color_score < 5000:  # Very high threshold
                return False
            
            # Method 2: Brightness analysis - must be very bright
            brightness = np.mean(head_region)
            if brightness < 180:  # High threshold
                return False
            
            # Method 3: Shape analysis - must be very helmet-like
            gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            helmet_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # High threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.8 <= aspect_ratio <= 1.5:  # Helmet-like ratio
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.6:  # Must be quite circular
                                helmet_shapes += 1
            
            # Must have multiple helmet-like shapes
            if helmet_shapes < 2:  # High threshold
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in strict helmet detection: {e}")
            return False
    
    def _detect_vest_strict(self, torso_region: np.ndarray) -> bool:
        """
        STRICT vest detection - only detects obvious bright vests
        
        Args:
            torso_region: Image region containing the torso
            
        Returns:
            True if vest is detected with high confidence
        """
        if torso_region.size == 0:
            return False
        
        try:
            # Method 1: STRICT color detection - only very bright colors
            hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            
            # Only very bright, obvious vest colors
            vest_colors = [
                ([15, 220, 220], [35, 255, 255]),  # Very bright yellow
                ([20, 180, 250], [30, 255, 255]),   # Very bright high-vis yellow
                ([10, 220, 220], [25, 255, 255]),   # Very bright orange
                ([40, 220, 220], [80, 255, 255]),   # Very bright green
                ([0, 220, 220], [10, 255, 255]),    # Very bright red
                ([0, 0, 250], [180, 20, 255])       # Very bright white
            ]
            
            color_score = 0
            for lower, upper in vest_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = np.sum(mask)
                if pixel_count > 3000:  # Very high threshold
                    color_score += pixel_count
            
            # Must have very significant bright color presence
            if color_score < 8000:  # Very high threshold
                return False
            
            # Method 2: Brightness and contrast analysis - must be very bright
            brightness = np.mean(torso_region)
            contrast = np.std(torso_region)
            
            if brightness < 200 or contrast < 80:  # Very high thresholds
                return False
            
            # Method 3: Strip detection - must have clear reflective strips
            gray = cv2.cvtColor(torso_region, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            edges = cv2.Canny(equalized, 20, 80)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            strips = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # High threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if aspect_ratio > 2.5 or aspect_ratio < 0.3:  # Clear strip-like
                        strips += 1
            
            # Must have multiple clear strips
            if strips < 3:  # High threshold
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in strict vest detection: {e}")
            return False
    
    def _get_missing_ppe(self, helmet_detected: bool, vest_detected: bool) -> List[str]:
        """Get list of missing PPE items"""
        missing = []
        if not helmet_detected:
            missing.append('helmet')
        if not vest_detected:
            missing.append('vest')
        return missing
    
    def _calculate_compliance(self, detections: List[Dict]) -> Dict[str, Any]:
        """Calculate overall PPE compliance statistics"""
        if not detections:
            return {
                'compliance_rate': 0.0,
                'total_persons': 0,
                'compliant_persons': 0,
                'missing_helmets': 0,
                'missing_vests': 0
            }
        
        total_persons = len(detections)
        compliant_persons = sum(1 for det in detections if det['ppe_compliant'])
        missing_helmets = sum(1 for det in detections if not det['helmet_detected'])
        missing_vests = sum(1 for det in detections if not det['vest_detected'])
        
        return {
            'compliance_rate': (compliant_persons / total_persons) * 100,
            'total_persons': total_persons,
            'compliant_persons': compliant_persons,
            'missing_helmets': missing_helmets,
            'missing_vests': missing_vests
        }
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Enhanced drawing with clear flagging of missing PPE
        
        Args:
            image: Input image
            detections: Detection results
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['person_bbox']
            
            # Determine person status color
            if detection['ppe_compliant']:
                person_color = self.colors['safe']
                status_text = "SAFE"
            else:
                person_color = self.colors['unsafe']
                status_text = "UNSAFE"
            
            # Draw person bounding box with status color
            cv2.rectangle(result_image, (x1, y1), (x2, y2), person_color, 3)
            
            # Draw PPE status indicators
            y_offset = y1 - 10 if y1 > 30 else y2 + 20
            
            # Helmet status with color coding
            if detection['helmet_detected']:
                helmet_color = self.colors['helmet']
                helmet_text = "✓ HELMET"
            else:
                helmet_color = self.colors['missing_helmet']
                helmet_text = "✗ NO HELMET"
            
            cv2.putText(result_image, helmet_text, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, helmet_color, 2)
            
            # Vest status with color coding
            if detection['vest_detected']:
                vest_color = self.colors['vest']
                vest_text = "✓ VEST"
            else:
                vest_color = self.colors['missing_vest']
                vest_text = "✗ NO VEST"
            
            cv2.putText(result_image, vest_text, (x1, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, vest_color, 2)
            
            # Overall safety status
            cv2.putText(result_image, status_text, (x1, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, person_color, 3)
            
            # Add missing PPE details
            if detection['missing_ppe']:
                missing_text = f"MISSING: {', '.join(detection['missing_ppe']).upper()}"
                cv2.putText(result_image, missing_text, (x1, y_offset + 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['unsafe'], 2)
        
        return result_image
