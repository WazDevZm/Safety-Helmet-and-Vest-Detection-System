"""
Safety Helmet and Vest Detection System - Simplified Version
Core detection module using YOLOv8 for PPE detection (without matplotlib dependencies)
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
    
    def _analyze_person_ppe(self, image: np.ndarray, person: Dict) -> Dict:
        """
        Analyze a detected person for PPE compliance with improved region detection
        
        Args:
            image: Input image
            person: Person detection data
            
        Returns:
            Person analysis with PPE status
        """
        x1, y1, x2, y2 = person['bbox']
        
        # Extract person region with padding for better detection
        padding = 10
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(image.shape[1], x2 + padding)
        y2_padded = min(image.shape[0], y2 + padding)
        
        person_region = image[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Improved region analysis for helmet (head and upper neck area)
        head_height = int(person_region.shape[0] * 0.35)  # Slightly smaller for more precise head detection
        helmet_region = person_region[:head_height, :]
        helmet_detected = self._detect_helmet(helmet_region)
        
        # Improved region analysis for vest (chest and torso area)
        vest_start = int(person_region.shape[0] * 0.15)  # Start from upper chest
        vest_end = int(person_region.shape[0] * 0.85)   # End at lower torso
        vest_region = person_region[vest_start:vest_end, :]
        vest_detected = self._detect_vest(vest_region)
        
        # Additional validation: check if regions are large enough for analysis
        min_region_size = 20  # Minimum pixels for meaningful analysis
        
        if helmet_region.shape[0] < min_region_size or helmet_region.shape[1] < min_region_size:
            helmet_detected = False  # Region too small for reliable detection
        
        if vest_region.shape[0] < min_region_size or vest_region.shape[1] < min_region_size:
            vest_detected = False  # Region too small for reliable detection
        
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
        Detect helmet in the head region using enhanced color and shape analysis
        
        Args:
            head_region: Image region containing the head
            
        Returns:
            True if helmet is detected
        """
        if head_region.size == 0:
            return False
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            
            # Enhanced color ranges for common helmet colors
            helmet_colors = [
                # Yellow/Orange helmets (most common)
                ([15, 100, 100], [35, 255, 255]),
                # White helmets
                ([0, 0, 180], [180, 30, 255]),
                # Red helmets
                ([0, 100, 100], [10, 255, 255]),
                # Blue helmets
                ([100, 100, 100], [130, 255, 255]),
                # Green helmets
                ([40, 100, 100], [80, 255, 255]),
                # Orange helmets
                ([10, 150, 150], [25, 255, 255])
            ]
            
            # Check for helmet-like colors with multiple thresholds
            color_score = 0
            for lower, upper in helmet_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = np.sum(mask)
                if pixel_count > 500:  # Lower threshold for better detection
                    color_score += pixel_count
            
            # If we have significant color presence, likely a helmet
            if color_score > 2000:
                return True
            
            # Enhanced shape analysis for hard hat detection
            gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive thresholding for better edge detection
            edges = cv2.Canny(blurred, 30, 100)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for helmet-like shapes
            helmet_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Lower area threshold
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check for helmet-like characteristics
                    if 0.7 <= aspect_ratio <= 2.0:  # Wider range for helmet shapes
                        # Check if it's roughly circular or oval
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.3:  # Somewhat circular
                                helmet_shapes += 1
            
            # If we found multiple helmet-like shapes, likely a helmet
            if helmet_shapes >= 2:
                return True
            
            # Additional texture analysis for hard hat material
            # Hard hats often have a distinctive texture
            texture_score = self._analyze_texture(head_region)
            if texture_score > 0.3:
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in helmet detection: {e}")
            return False
    
    def _detect_vest(self, torso_region: np.ndarray) -> bool:
        """
        Detect reflective vest in the torso region using enhanced algorithms
        
        Args:
            torso_region: Image region containing the torso
            
        Returns:
            True if vest is detected
        """
        if torso_region.size == 0:
            return False
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
            
            # Enhanced color ranges for reflective vests
            vest_colors = [
                # Bright yellow/orange (most common for safety vests)
                ([15, 120, 120], [35, 255, 255]),
                # High visibility yellow
                ([20, 80, 180], [30, 255, 255]),
                # Bright orange
                ([10, 120, 120], [25, 255, 255]),
                # Lime green safety vests
                ([40, 100, 100], [80, 255, 255]),
                # Red safety vests
                ([0, 100, 100], [10, 255, 255]),
                # White reflective vests
                ([0, 0, 200], [180, 30, 255])
            ]
            
            # Check for vest-like colors with scoring system
            color_score = 0
            for lower, upper in vest_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = np.sum(mask)
                if pixel_count > 1000:  # Lower threshold for better detection
                    color_score += pixel_count
            
            # If we have significant bright color presence, likely a vest
            if color_score > 3000:
                return True
            
            # Enhanced texture analysis for reflective material
            gray = cv2.cvtColor(torso_region, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to enhance contrast
            equalized = cv2.equalizeHist(gray)
            
            # Look for high contrast areas (reflective strips)
            edges = cv2.Canny(equalized, 20, 80)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for horizontal strips (common in safety vests)
            horizontal_strips = 0
            vertical_strips = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Lower threshold for better detection
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if aspect_ratio > 2:  # Horizontal strips
                        horizontal_strips += 1
                    elif aspect_ratio < 0.5:  # Vertical strips
                        vertical_strips += 1
            
            # Safety vests often have both horizontal and vertical reflective strips
            if horizontal_strips >= 1 and vertical_strips >= 1:
                return True
            elif horizontal_strips >= 3:  # Many horizontal strips
                return True
            elif vertical_strips >= 2:  # Multiple vertical strips
                return True
            
            # Additional analysis for reflective material properties
            # Reflective vests often have high brightness and contrast
            brightness = np.mean(torso_region)
            contrast = np.std(torso_region)
            
            # High brightness and contrast might indicate reflective material
            if brightness > 150 and contrast > 50:
                return True
            
            # Check for vest-like shape (rectangular coverage)
            vest_coverage = self._analyze_vest_coverage(torso_region)
            if vest_coverage > 0.4:  # Good coverage of torso area
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in vest detection: {e}")
            return False
    
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
    
    def _analyze_texture(self, region: np.ndarray) -> float:
        """
        Analyze texture characteristics for hard hat material detection
        
        Args:
            region: Image region to analyze
            
        Returns:
            Texture score (0-1)
        """
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Calculate local binary pattern-like features
            # Hard hats often have distinctive surface texture
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate texture score based on gradient patterns
            texture_score = np.mean(gradient_magnitude) / 255.0
            
            return min(texture_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _analyze_vest_coverage(self, region: np.ndarray) -> float:
        """
        Analyze vest coverage of the torso region
        
        Args:
            region: Torso region to analyze
            
        Returns:
            Coverage score (0-1)
        """
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define bright colors that might indicate vest coverage
            bright_colors = [
                ([15, 100, 100], [35, 255, 255]),  # Yellow/Orange
                ([0, 0, 200], [180, 30, 255]),     # White
                ([0, 100, 100], [10, 255, 255]),   # Red
                ([40, 100, 100], [80, 255, 255])   # Green
            ]
            
            total_coverage = 0
            for lower, upper in bright_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                coverage = np.sum(mask) / (region.shape[0] * region.shape[1] * 255)
                total_coverage += coverage
            
            return min(total_coverage, 1.0)
            
        except Exception:
            return 0.0
