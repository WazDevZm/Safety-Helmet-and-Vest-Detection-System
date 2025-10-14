"""
Ore Image Preprocessing Pipeline
Advanced image preprocessing for ore quality classification
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any
import os
from pathlib import Path
import json

class OrePreprocessor:
    """
    Advanced image preprocessing for ore samples
    Handles various image enhancement and feature extraction techniques
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the ore preprocessor
        
        Args:
            target_size: Target image size (width, height)
        """
        self.target_size = target_size
        self.enhancement_params = {
            'gamma': 1.2,
            'alpha': 1.1,
            'beta': 10,
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': (8, 8)
        }
    
    def preprocess_image(self, image_path: str, enhancement: bool = True) -> Dict[str, np.ndarray]:
        """
        Comprehensive image preprocessing pipeline
        
        Args:
            image_path: Path to input image
            enhancement: Whether to apply enhancement techniques
            
        Returns:
            Dictionary containing processed images and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Basic preprocessing
        processed = self._basic_preprocessing(image_rgb)
        
        # Enhancement if requested
        if enhancement:
            enhanced = self._enhance_image(processed)
        else:
            enhanced = processed
        
        # Resize to target size
        resized = cv2.resize(enhanced, self.target_size)
        
        # Extract features
        features = self._extract_visual_features(resized)
        
        # Create masks for different ore components
        masks = self._create_ore_masks(resized)
        
        return {
            'original': image_rgb,
            'processed': processed,
            'enhanced': enhanced,
            'resized': resized,
            'features': features,
            'masks': masks,
            'metadata': self._extract_metadata(image_path, resized)
        }
    
    def _basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Basic image preprocessing steps
        
        Args:
            image: Input RGB image
            
        Returns:
            Preprocessed image
        """
        # Noise reduction
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Normalize intensity
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized.astype(np.uint8)
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced image enhancement for ore samples
        
        Args:
            image: Preprocessed image
            
        Returns:
            Enhanced image
        """
        # Gamma correction
        gamma_corrected = self._gamma_correction(image, self.enhancement_params['gamma'])
        
        # Contrast enhancement
        contrast_enhanced = self._contrast_enhancement(
            gamma_corrected, 
            self.enhancement_params['alpha'], 
            self.enhancement_params['beta']
        )
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_enhanced = self._apply_clahe(contrast_enhanced)
        
        # Color space enhancement
        color_enhanced = self._enhance_color_space(clahe_enhanced)
        
        return color_enhanced
    
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def _contrast_enhancement(self, image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
        """Apply contrast enhancement"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to enhance local contrast"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.enhancement_params['clahe_clip_limit'],
            tileGridSize=self.enhancement_params['clahe_tile_size']
        )
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    
    def _enhance_color_space(self, image: np.ndarray) -> np.ndarray:
        """Enhance color space for better ore feature detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance saturation
        s = cv2.multiply(s, 1.2)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Enhance value (brightness)
        v = cv2.multiply(v, 1.1)
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        return enhanced_rgb
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive visual features from ore image
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Color features
        color_features = self._extract_color_features(image)
        features.update(color_features)
        
        # Texture features
        texture_features = self._extract_texture_features(image)
        features.update(texture_features)
        
        # Shape features
        shape_features = self._extract_shape_features(image)
        features.update(shape_features)
        
        # Surface features
        surface_features = self._extract_surface_features(image)
        features.update(surface_features)
        
        return features
    
    def _extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-based features"""
        features = {}
        
        # RGB statistics
        r, g, b = cv2.split(image)
        features['mean_red'] = float(np.mean(r))
        features['mean_green'] = float(np.mean(g))
        features['mean_blue'] = float(np.mean(b))
        features['std_red'] = float(np.std(r))
        features['std_green'] = float(np.std(g))
        features['std_blue'] = float(np.std(b))
        
        # HSV statistics
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        features['mean_hue'] = float(np.mean(h))
        features['mean_saturation'] = float(np.mean(s))
        features['mean_value'] = float(np.mean(v))
        features['std_hue'] = float(np.std(h))
        features['std_saturation'] = float(np.std(s))
        features['std_value'] = float(np.std(v))
        
        # Color diversity
        features['color_diversity'] = float(len(np.unique(image.reshape(-1, 3), axis=0)))
        
        # Dominant colors
        dominant_colors = self._get_dominant_colors(image, k=5)
        for i, color in enumerate(dominant_colors):
            features[f'dominant_color_{i}_r'] = float(color[0])
            features[f'dominant_color_{i}_g'] = float(color[1])
            features[f'dominant_color_{i}_b'] = float(color[2])
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture-based features"""
        features = {}
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GLCM features
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Calculate GLCM
            glcm = graycomatrix(gray, distances=[1, 2], angles=[0, 45, 90, 135], 
                              levels=256, symmetric=True, normed=True)
            
            # Extract texture properties
            features['contrast'] = float(np.mean(graycoprops(glcm, 'contrast')))
            features['dissimilarity'] = float(np.mean(graycoprops(glcm, 'dissimilarity')))
            features['homogeneity'] = float(np.mean(graycoprops(glcm, 'homogeneity')))
            features['energy'] = float(np.mean(graycoprops(glcm, 'energy')))
            features['correlation'] = float(np.mean(graycoprops(glcm, 'correlation')))
            features['asm'] = float(np.mean(graycoprops(glcm, 'ASM')))
            
        except ImportError:
            # Fallback texture features
            features['contrast'] = float(np.std(gray))
            features['dissimilarity'] = float(np.mean(np.abs(np.diff(gray.flatten()))))
            features['homogeneity'] = float(1.0 / (1.0 + np.var(gray)))
            features['energy'] = float(np.sum(gray**2) / (gray.shape[0] * gray.shape[1]))
            features['correlation'] = 0.0
            features['asm'] = 0.0
        
        # Additional texture features
        features['texture_std'] = float(np.std(gray))
        features['texture_mean'] = float(np.mean(gray))
        features['texture_skewness'] = float(self._calculate_skewness(gray))
        features['texture_kurtosis'] = float(self._calculate_kurtosis(gray))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        return features
    
    def _extract_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features"""
        features = {}
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['contour_area'] = float(area)
            features['contour_perimeter'] = float(perimeter)
            features['circularity'] = float(4 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            features['aspect_ratio'] = float(w / h) if h > 0 else 0.0
            features['extent'] = float(area / (w * h)) if w * h > 0 else 0.0
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = float(area / hull_area) if hull_area > 0 else 0.0
            
        else:
            # Default values if no contours found
            features.update({
                'contour_area': 0.0,
                'contour_perimeter': 0.0,
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'extent': 0.0,
                'solidity': 0.0
            })
        
        return features
    
    def _extract_surface_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract surface characteristics"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Surface roughness indicators
        features['surface_roughness'] = float(np.std(gray))
        features['surface_smoothness'] = float(1.0 / (1.0 + np.std(gray)))
        
        # Brightness and contrast
        features['brightness'] = float(np.mean(gray))
        features['contrast'] = float(np.std(gray))
        
        # Local binary pattern (simplified)
        features['lbp_uniformity'] = float(self._calculate_lbp_uniformity(gray))
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = float(np.mean(gradient_magnitude))
        features['gradient_std'] = float(np.std(gradient_magnitude))
        
        return features
    
    def _create_ore_masks(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create masks for different ore components
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of masks for different ore types
        """
        masks = {}
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color ranges for different ore types
        ore_ranges = {
            'copper': ([10, 50, 50], [20, 255, 255]),
            'iron': ([0, 50, 50], [10, 255, 255]),
            'gold': ([20, 50, 50], [30, 255, 255]),
            'silver': ([0, 0, 100], [180, 30, 255]),
            'high_grade': ([0, 100, 100], [180, 255, 255])
        }
        
        for ore_type, (lower, upper) in ore_ranges.items():
            # Create mask
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Apply morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            masks[ore_type] = mask
        
        return masks
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[np.ndarray]:
        """Extract dominant colors using K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape image to list of pixels
            pixels = image.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (dominant colors)
            dominant_colors = kmeans.cluster_centers_.astype(int)
            
            return dominant_colors.tolist()
            
        except ImportError:
            # Fallback: simple color sampling
            step = max(1, len(image) // k)
            colors = []
            for i in range(0, len(image), step):
                for j in range(0, len(image[i]), step):
                    colors.append(image[i, j])
                    if len(colors) >= k:
                        break
                if len(colors) >= k:
                    break
            
            return colors[:k]
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4)) - 3.0
    
    def _calculate_lbp_uniformity(self, image: np.ndarray) -> float:
        """Calculate LBP uniformity (simplified)"""
        # Simplified LBP calculation
        rows, cols = image.shape
        lbp_image = np.zeros_like(image)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = image[i, j]
                binary_string = ""
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if image[i+di, j+dj] >= center:
                            binary_string += "1"
                        else:
                            binary_string += "0"
                
                # Count transitions
                transitions = sum(1 for k in range(len(binary_string)) 
                                if binary_string[k] != binary_string[(k+1) % len(binary_string)])
                
                lbp_image[i, j] = transitions
        
        return float(np.mean(lbp_image))
    
    def _extract_metadata(self, image_path: str, image: np.ndarray) -> Dict[str, Any]:
        """Extract metadata from image"""
        return {
            'filename': os.path.basename(image_path),
            'file_size': os.path.getsize(image_path),
            'image_shape': image.shape,
            'image_dtype': str(image.dtype),
            'total_pixels': image.size
        }
    
    def batch_preprocess(self, input_dir: str, output_dir: str, 
                        enhancement: bool = True) -> Dict[str, Any]:
        """
        Batch process multiple images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images
            enhancement: Whether to apply enhancement
            
        Returns:
            Processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        processed_count = 0
        error_count = 0
        errors = []
        
        # Process all images in directory
        for image_file in input_path.iterdir():
            if image_file.suffix.lower() in extensions:
                try:
                    # Process image
                    result = self.preprocess_image(str(image_file), enhancement)
                    
                    # Save processed image
                    output_file = output_path / f"processed_{image_file.name}"
                    cv2.imwrite(str(output_file), cv2.cvtColor(result['enhanced'], cv2.COLOR_RGB2BGR))
                    
                    # Save features as JSON
                    features_file = output_path / f"features_{image_file.stem}.json"
                    with open(features_file, 'w') as f:
                        json.dump(result['features'], f, indent=2)
                    
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    errors.append(f"{image_file.name}: {str(e)}")
        
        return {
            'processed_count': processed_count,
            'error_count': error_count,
            'errors': errors,
            'success_rate': processed_count / (processed_count + error_count) if (processed_count + error_count) > 0 else 0
        }

