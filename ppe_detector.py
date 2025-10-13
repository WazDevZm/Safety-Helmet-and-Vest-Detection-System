"""
Safety Helmet and Vest Detection System
Core detection module using YOLOv8 for PPE detection
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Tuple, Dict, Any
import os

class PPEDetector:
    """
    Personal Protective Equipment (PPE) Detection System
    Detects safety helmets and reflective vests in images
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the PPE detector
        
        Args:
            model_path: Path to custom YOLOv8 model (optional)
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLOv8 model - using pre-trained COCO model as base
        # In production, you would train a custom model on PPE datasets
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Using YOLOv8n (nano) for faster inference
            self.model = YOLO('yolov8n.pt')
        
        # PPE class mappings (based on COCO dataset)
        self.ppe_classes = {
            'person': 0,  # We'll detect persons and then check for PPE
            'helmet': 'helmet',  # Custom class for helmets
            'vest': 'vest'  # Custom class for safety vests
        }
        
        # Colors for bounding boxes
        self.colors = {
            'helmet': (0, 255, 0),      # Green for helmet
            'vest': (255, 0, 0),        # Blue for vest
            'missing_helmet': (0, 0, 255),  # Red for missing helmet
            'missing_vest': (0, 165, 255)   # Orange for missing vest
        }
    
    def detect_ppe(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect PPE in the given image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        # Run YOLOv8 inference
        results = self.model(image, verbose=False)
        
        # Process results
        detections = []
        persons_detected = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only process person detections
                    if class_id == 0 and confidence > 0.5:  # Person class
                        person_bbox = [int(x1), int(y1), int(x2), int(y2)]
                        persons_detected.append({
                            'bbox': person_bbox,
                            'confidence': confidence
                        })
        
        # Analyze each detected person for PPE
        for person in persons_detected:
            person_analysis = self._analyze_person_ppe(image, person)
            detections.append(person_analysis)
        
        return {
            'detections': detections,
            'total_persons': len(persons_detected),
            'ppe_compliance': self._calculate_compliance(detections)
        }
    
    def _analyze_person_ppe(self, image: np.ndarray, person: Dict) -> Dict:
        """
        Analyze a detected person for PPE compliance
        
        Args:
            image: Input image
            person: Person detection data
            
        Returns:
            Person analysis with PPE status
        """
        x1, y1, x2, y2 = person['bbox']
        
        # Extract person region
        person_region = image[y1:y2, x1:x2]
        
        # Analyze for helmet (upper body region)
        helmet_region = person_region[:int(person_region.shape[0] * 0.4), :]
        helmet_detected = self._detect_helmet(helmet_region)
        
        # Analyze for vest (torso region)
        vest_region = person_region[int(person_region.shape[0] * 0.2):int(person_region.shape[0] * 0.8), :]
        vest_detected = self._detect_vest(vest_region)
        
        return {
            'person_bbox': person['bbox'],
            'confidence': person['confidence'],
            'helmet_detected': helmet_detected,
            'vest_detected': vest_detected,
            'ppe_compliant': helmet_detected and vest_detected,
            'missing_ppe': self._get_missing_ppe(helmet_detected, vest_detected)
        }
    
    def _detect_helmet(self, head_region: np.ndarray) -> bool:
        """
        Detect helmet in the head region using color and shape analysis
        
        Args:
            head_region: Image region containing the head
            
        Returns:
            True if helmet is detected
        """
        if head_region.size == 0:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common helmet colors
        helmet_colors = [
            # Yellow/Orange helmets
            ([20, 100, 100], [30, 255, 255]),
            # White helmets
            ([0, 0, 200], [180, 30, 255]),
            # Red helmets
            ([0, 100, 100], [10, 255, 255]),
            # Blue helmets
            ([100, 100, 100], [130, 255, 255])
        ]
        
        # Check for helmet-like colors
        for lower, upper in helmet_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask) > 1000:  # Threshold for color presence
                return True
        
        # Additional shape analysis for hard hat detection
        gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for helmet-like shapes (circular or oval)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                # Check aspect ratio (helmets are typically wider than tall)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.8 <= aspect_ratio <= 1.5:  # Reasonable helmet aspect ratio
                    return True
        
        return False
    
    def _detect_vest(self, torso_region: np.ndarray) -> bool:
        """
        Detect reflective vest in the torso region
        
        Args:
            torso_region: Image region containing the torso
            
        Returns:
            True if vest is detected
        """
        if torso_region.size == 0:
            return False
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for reflective vests (bright yellow/orange)
        vest_colors = [
            # Bright yellow/orange (most common for safety vests)
            ([15, 150, 150], [35, 255, 255]),
            # High visibility yellow
            ([20, 100, 200], [30, 255, 255]),
            # Bright orange
            ([10, 150, 150], [25, 255, 255])
        ]
        
        # Check for vest-like colors
        for lower, upper in vest_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if np.sum(mask) > 2000:  # Higher threshold for vest detection
                return True
        
        # Additional texture analysis for reflective material
        gray = cv2.cvtColor(torso_region, cv2.COLOR_BGR2GRAY)
        
        # Look for high contrast areas (reflective strips)
        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for horizontal strips (common in safety vests)
        horizontal_strips = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if aspect_ratio > 2:  # Horizontal strips
                    horizontal_strips += 1
        
        return horizontal_strips >= 2  # At least 2 horizontal strips detected
    
    def _get_missing_ppe(self, helmet_detected: bool, vest_detected: bool) -> List[str]:
        """
        Get list of missing PPE items
        
        Args:
            helmet_detected: Whether helmet was detected
            vest_detected: Whether vest was detected
            
        Returns:
            List of missing PPE items
        """
        missing = []
        if not helmet_detected:
            missing.append('helmet')
        if not vest_detected:
            missing.append('vest')
        return missing
    
    def _calculate_compliance(self, detections: List[Dict]) -> Dict[str, Any]:
        """
        Calculate overall PPE compliance statistics
        
        Args:
            detections: List of person detections with PPE analysis
            
        Returns:
            Compliance statistics
        """
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
        Draw detection results on the image
        
        Args:
            image: Input image
            detections: Detection results
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['person_bbox']
            
            # Draw person bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Draw PPE status indicators
            y_offset = y1 - 10 if y1 > 30 else y2 + 20
            
            # Helmet status
            helmet_color = self.colors['helmet'] if detection['helmet_detected'] else self.colors['missing_helmet']
            helmet_text = "✓ Helmet" if detection['helmet_detected'] else "✗ No Helmet"
            cv2.putText(result_image, helmet_text, (x1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, helmet_color, 2)
            
            # Vest status
            vest_color = self.colors['vest'] if detection['vest_detected'] else self.colors['missing_vest']
            vest_text = "✓ Vest" if detection['vest_detected'] else "✗ No Vest"
            cv2.putText(result_image, vest_text, (x1, y_offset + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vest_color, 2)
            
            # Overall compliance status
            if detection['ppe_compliant']:
                cv2.putText(result_image, "SAFE", (x1, y_offset + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(result_image, "UNSAFE", (x1, y_offset + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result_image
