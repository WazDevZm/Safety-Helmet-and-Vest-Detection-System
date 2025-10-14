"""
Ore Training Data Generation and Augmentation System
Creates synthetic ore samples and augments existing data for training
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

@dataclass
class OreSample:
    """Data class for ore sample information"""
    image_path: str
    quality_grade: str
    mineral_type: str
    features: Dict[str, Any]
    metadata: Dict[str, Any]

class OreDataGenerator:
    """
    Advanced data generation and augmentation for ore classification
    Creates synthetic ore samples and augments existing datasets
    """
    
    def __init__(self, output_dir: str = "ore_dataset"):
        """
        Initialize the ore data generator
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality grades and their characteristics
        self.quality_grades = {
            'Very High Grade': {
                'color_intensity': (0.8, 1.0),
                'texture_complexity': (0.7, 1.0),
                'surface_roughness': (0.3, 0.6),
                'brightness': (0.7, 1.0),
                'contrast': (0.6, 1.0)
            },
            'High Grade': {
                'color_intensity': (0.6, 0.9),
                'texture_complexity': (0.5, 0.8),
                'surface_roughness': (0.4, 0.7),
                'brightness': (0.6, 0.9),
                'contrast': (0.5, 0.9)
            },
            'Medium Grade': {
                'color_intensity': (0.4, 0.7),
                'texture_complexity': (0.3, 0.6),
                'surface_roughness': (0.5, 0.8),
                'brightness': (0.4, 0.7),
                'contrast': (0.4, 0.8)
            },
            'Low Grade': {
                'color_intensity': (0.2, 0.5),
                'texture_complexity': (0.2, 0.5),
                'surface_roughness': (0.6, 0.9),
                'brightness': (0.3, 0.6),
                'contrast': (0.3, 0.7)
            },
            'Very Low Grade': {
                'color_intensity': (0.1, 0.4),
                'texture_complexity': (0.1, 0.4),
                'surface_roughness': (0.7, 1.0),
                'brightness': (0.2, 0.5),
                'contrast': (0.2, 0.6)
            }
        }
        
        # Mineral types and their base colors
        self.mineral_types = {
            'Copper': {
                'base_colors': [(0, 100, 200), (20, 120, 220), (40, 140, 240)],
                'texture_patterns': ['granular', 'massive', 'vein']
            },
            'Iron': {
                'base_colors': [(50, 50, 50), (80, 80, 80), (120, 120, 120)],
                'texture_patterns': ['banded', 'massive', 'granular']
            },
            'Gold': {
                'base_colors': [(255, 215, 0), (255, 200, 0), (255, 180, 0)],
                'texture_patterns': ['vein', 'nugget', 'placer']
            },
            'Silver': {
                'base_colors': [(192, 192, 192), (200, 200, 200), (220, 220, 220)],
                'texture_patterns': ['wire', 'massive', 'granular']
            },
            'Lead': {
                'base_colors': [(100, 100, 100), (120, 120, 120), (140, 140, 140)],
                'texture_patterns': ['massive', 'granular', 'cubic']
            }
        }
    
    def generate_synthetic_ore_samples(self, num_samples: int = 1000, 
                                      image_size: Tuple[int, int] = (224, 224)) -> List[OreSample]:
        """
        Generate synthetic ore samples with various quality grades
        
        Args:
            num_samples: Number of samples to generate
            image_size: Size of generated images
            
        Returns:
            List of generated ore samples
        """
        samples = []
        
        # Create subdirectories for each quality grade
        for grade in self.quality_grades.keys():
            grade_dir = self.output_dir / grade
            grade_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            # Randomly select quality grade and mineral type
            quality_grade = random.choice(list(self.quality_grades.keys()))
            mineral_type = random.choice(list(self.mineral_types.keys()))
            
            # Generate synthetic ore image
            ore_image = self._create_synthetic_ore_image(
                quality_grade, mineral_type, image_size
            )
            
            # Save image
            filename = f"synthetic_{mineral_type}_{quality_grade}_{i:04d}.jpg"
            image_path = self.output_dir / quality_grade / filename
            
            # Convert RGB to BGR for OpenCV
            ore_image_bgr = cv2.cvtColor(ore_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_path), ore_image_bgr)
            
            # Extract features
            features = self._extract_ore_features(ore_image)
            
            # Create metadata
            metadata = {
                'synthetic': True,
                'generation_method': 'procedural',
                'quality_grade': quality_grade,
                'mineral_type': mineral_type,
                'image_size': image_size,
                'generation_params': self.quality_grades[quality_grade]
            }
            
            # Create ore sample
            sample = OreSample(
                image_path=str(image_path),
                quality_grade=quality_grade,
                mineral_type=mineral_type,
                features=features,
                metadata=metadata
            )
            
            samples.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        # Save dataset metadata
        self._save_dataset_metadata(samples)
        
        return samples
    
    def _create_synthetic_ore_image(self, quality_grade: str, mineral_type: str, 
                                   image_size: Tuple[int, int]) -> np.ndarray:
        """
        Create a synthetic ore image based on quality grade and mineral type
        
        Args:
            quality_grade: Quality grade of the ore
            mineral_type: Type of mineral
            image_size: Size of the generated image
            
        Returns:
            Generated ore image
        """
        height, width = image_size
        
        # Get quality parameters
        quality_params = self.quality_grades[quality_grade]
        mineral_info = self.mineral_types[mineral_type]
        
        # Create base image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Generate base color
        base_color = random.choice(mineral_info['base_colors'])
        
        # Apply quality-based modifications
        color_intensity = random.uniform(*quality_params['color_intensity'])
        brightness = random.uniform(*quality_params['brightness'])
        contrast = random.uniform(*quality_params['contrast'])
        
        # Create base color with quality modifications
        base_r, base_g, base_b = base_color
        modified_color = (
            int(base_r * color_intensity * brightness),
            int(base_g * color_intensity * brightness),
            int(base_b * color_intensity * brightness)
        )
        
        # Fill base color
        image[:] = modified_color
        
        # Add texture based on quality
        texture_complexity = random.uniform(*quality_params['texture_complexity'])
        surface_roughness = random.uniform(*quality_params['surface_roughness'])
        
        # Generate texture
        image = self._add_ore_texture(image, texture_complexity, surface_roughness)
        
        # Add mineral-specific patterns
        pattern_type = random.choice(mineral_info['texture_patterns'])
        image = self._add_mineral_pattern(image, pattern_type, quality_grade)
        
        # Add noise and imperfections
        image = self._add_ore_imperfections(image, quality_grade)
        
        # Apply final quality-based adjustments
        image = self._apply_quality_adjustments(image, quality_params)
        
        return image
    
    def _add_ore_texture(self, image: np.ndarray, complexity: float, roughness: float) -> np.ndarray:
        """Add texture to ore image"""
        height, width, channels = image.shape
        
        # Generate noise patterns
        noise_scale = int(roughness * 20) + 1
        
        # Perlin-like noise for natural texture
        for c in range(channels):
            # Create multiple noise layers
            noise = np.zeros((height, width), dtype=np.float32)
            
            for scale in [1, 2, 4, 8]:
                if scale <= complexity * 8:
                    # Generate noise at different scales
                    noise_layer = np.random.normal(0, 1, (height // scale, width // scale))
                    noise_layer = cv2.resize(noise_layer, (width, height))
                    noise += noise_layer / scale
            
            # Apply noise to image
            noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
            noise = (noise - 0.5) * roughness * 50
            
            image[:, :, c] = np.clip(image[:, :, c] + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _add_mineral_pattern(self, image: np.ndarray, pattern_type: str, quality_grade: str) -> np.ndarray:
        """Add mineral-specific patterns"""
        height, width, channels = image.shape
        
        if pattern_type == 'vein':
            # Add vein patterns
            num_veins = random.randint(2, 8)
            for _ in range(num_veins):
                # Create vein
                start_x = random.randint(0, width)
                start_y = random.randint(0, height)
                end_x = random.randint(0, width)
                end_y = random.randint(0, height)
                
                # Draw vein
                cv2.line(image, (start_x, start_y), (end_x, end_y), 
                        (255, 255, 255), random.randint(2, 8))
        
        elif pattern_type == 'granular':
            # Add granular texture
            num_grains = random.randint(50, 200)
            for _ in range(num_grains):
                center_x = random.randint(0, width)
                center_y = random.randint(0, height)
                radius = random.randint(3, 15)
                
                # Draw grain
                cv2.circle(image, (center_x, center_y), radius, 
                          (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), -1)
        
        elif pattern_type == 'banded':
            # Add banded structure
            num_bands = random.randint(3, 10)
            band_height = height // num_bands
            
            for i in range(num_bands):
                y_start = i * band_height
                y_end = min((i + 1) * band_height, height)
                
                # Create band with slight color variation
                band_color = (
                    random.randint(0, 50),
                    random.randint(0, 50),
                    random.randint(0, 50)
                )
                image[y_start:y_end, :] = np.clip(
                    image[y_start:y_end, :] + band_color, 0, 255
                ).astype(np.uint8)
        
        elif pattern_type == 'massive':
            # Add massive structure with large color variations
            num_regions = random.randint(3, 8)
            for _ in range(num_regions):
                # Create irregular region
                center_x = random.randint(0, width)
                center_y = random.randint(0, height)
                radius = random.randint(20, 80)
                
                # Create mask for region
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)
                
                # Apply color variation
                color_variation = (
                    random.randint(-30, 30),
                    random.randint(-30, 30),
                    random.randint(-30, 30)
                )
                
                for c in range(channels):
                    image[:, :, c] = np.where(
                        mask > 0,
                        np.clip(image[:, :, c] + color_variation[c], 0, 255),
                        image[:, :, c]
                    ).astype(np.uint8)
        
        return image
    
    def _add_ore_imperfections(self, image: np.ndarray, quality_grade: str) -> np.ndarray:
        """Add realistic ore imperfections"""
        height, width, channels = image.shape
        
        # Add scratches and cracks
        num_imperfections = random.randint(5, 20)
        for _ in range(num_imperfections):
            # Random line for scratch/crack
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            end_x = random.randint(0, width)
            end_y = random.randint(0, height)
            
            # Draw imperfection
            color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            cv2.line(image, (start_x, start_y), (end_x, end_y), color, random.randint(1, 3))
        
        # Add dust and particles
        num_particles = random.randint(10, 50)
        for _ in range(num_particles):
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)
            radius = random.randint(1, 5)
            
            # Draw particle
            color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
            cv2.circle(image, (center_x, center_y), radius, color, -1)
        
        # Add lighting variations
        # Create gradient mask
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        gradient = 1 - (distance / max_distance)
        
        # Apply lighting effect
        for c in range(channels):
            image[:, :, c] = np.clip(
                image[:, :, c] * (0.7 + 0.3 * gradient), 0, 255
            ).astype(np.uint8)
        
        return image
    
    def _apply_quality_adjustments(self, image: np.ndarray, quality_params: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Apply final quality-based adjustments"""
        # Adjust brightness
        brightness = random.uniform(*quality_params['brightness'])
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Adjust contrast
        contrast = random.uniform(*quality_params['contrast'])
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
        
        # Apply slight blur for lower quality
        if brightness < 0.6:
            blur_kernel = random.randint(1, 3)
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        
        return image
    
    def _extract_ore_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features from ore image"""
        features = {}
        
        # Color features
        mean_color = np.mean(image, axis=(0, 1))
        std_color = np.std(image, axis=(0, 1))
        
        features['mean_red'] = float(mean_color[0])
        features['mean_green'] = float(mean_color[1])
        features['mean_blue'] = float(mean_color[2])
        features['std_red'] = float(std_color[0])
        features['std_green'] = float(std_color[1])
        features['std_blue'] = float(std_color[2])
        
        # Texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features['texture_std'] = float(np.std(gray))
        features['texture_mean'] = float(np.mean(gray))
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        return features
    
    def augment_existing_dataset(self, input_dir: str, output_dir: str, 
                                augmentation_factor: int = 3) -> Dict[str, Any]:
        """
        Augment existing ore dataset with various transformations
        
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
            augmentation_factor: Number of augmented versions per original image
            
        Returns:
            Augmentation statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each quality grade
        for grade in self.quality_grades.keys():
            grade_dir = output_path / grade
            grade_dir.mkdir(exist_ok=True)
        
        augmented_count = 0
        original_count = 0
        
        # Process each quality grade directory
        for grade_dir in input_path.iterdir():
            if grade_dir.is_dir():
                grade_name = grade_dir.name
                output_grade_dir = output_path / grade_name
                
                # Process images in grade directory
                for image_file in grade_dir.iterdir():
                    if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        original_count += 1
                        
                        # Load original image
                        original_image = cv2.imread(str(image_file))
                        if original_image is None:
                            continue
                        
                        # Create augmented versions
                        for i in range(augmentation_factor):
                            augmented_image = self._augment_single_image(original_image)
                            
                            # Save augmented image
                            aug_filename = f"aug_{i}_{image_file.name}"
                            aug_path = output_grade_dir / aug_filename
                            cv2.imwrite(str(aug_path), augmented_image)
                            
                            augmented_count += 1
        
        return {
            'original_count': original_count,
            'augmented_count': augmented_count,
            'augmentation_factor': augmentation_factor,
            'total_generated': original_count + augmented_count
        }
    
    def _augment_single_image(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single image"""
        augmented = image.copy()
        
        # Random rotation
        if random.random() < 0.7:
            angle = random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))
        
        # Random flip
        if random.random() < 0.5:
            augmented = cv2.flip(augmented, 1)  # Horizontal flip
        
        # Random brightness/contrast adjustment
        if random.random() < 0.8:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-30, 30)   # Brightness
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
        
        # Random noise
        if random.random() < 0.6:
            noise = np.random.normal(0, 10, augmented.shape).astype(np.uint8)
            augmented = cv2.add(augmented, noise)
        
        # Random blur
        if random.random() < 0.3:
            blur_kernel = random.choice([3, 5, 7])
            augmented = cv2.GaussianBlur(augmented, (blur_kernel, blur_kernel), 0)
        
        # Random crop and resize
        if random.random() < 0.5:
            h, w = augmented.shape[:2]
            crop_size = random.randint(int(min(h, w) * 0.8), min(h, w))
            start_x = random.randint(0, w - crop_size)
            start_y = random.randint(0, h - crop_size)
            cropped = augmented[start_y:start_y + crop_size, start_x:start_x + crop_size]
            augmented = cv2.resize(cropped, (w, h))
        
        return augmented
    
    def _save_dataset_metadata(self, samples: List[OreSample]):
        """Save dataset metadata"""
        metadata = {
            'total_samples': len(samples),
            'quality_distribution': {},
            'mineral_distribution': {},
            'generation_info': {
                'generator_version': '1.0',
                'creation_date': str(Path().cwd()),
                'image_size': samples[0].metadata['image_size'] if samples else None
            }
        }
        
        # Calculate distributions
        for sample in samples:
            quality = sample.quality_grade
            mineral = sample.mineral_type
            
            metadata['quality_distribution'][quality] = metadata['quality_distribution'].get(quality, 0) + 1
            metadata['mineral_distribution'][mineral] = metadata['mineral_distribution'].get(mineral, 0) + 1
        
        # Save metadata
        metadata_path = self.output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset metadata saved to {metadata_path}")
    
    def create_train_test_split(self, data_dir: str, test_size: float = 0.2, 
                               random_state: int = 42) -> Dict[str, List[str]]:
        """
        Create train/test split for the dataset
        
        Args:
            data_dir: Directory containing the dataset
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train and test file lists
        """
        data_path = Path(data_dir)
        all_files = []
        
        # Collect all image files
        for grade_dir in data_path.iterdir():
            if grade_dir.is_dir():
                for image_file in grade_dir.iterdir():
                    if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        all_files.append(str(image_file))
        
        # Split into train and test
        train_files, test_files = train_test_split(
            all_files, test_size=test_size, random_state=random_state
        )
        
        # Save split information
        split_info = {
            'train_files': train_files,
            'test_files': test_files,
            'train_count': len(train_files),
            'test_count': len(test_files),
            'test_size': test_size,
            'random_state': random_state
        }
        
        split_path = data_path / 'train_test_split.json'
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Train/test split created: {len(train_files)} train, {len(test_files)} test")
        print(f"Split information saved to {split_path}")
        
        return split_info
    
    def visualize_dataset_statistics(self, data_dir: str, save_path: str = None):
        """
        Visualize dataset statistics and distributions
        
        Args:
            data_dir: Directory containing the dataset
            save_path: Path to save visualization
        """
        data_path = Path(data_dir)
        
        # Collect statistics
        quality_counts = {}
        mineral_counts = {}
        total_files = 0
        
        for grade_dir in data_path.iterdir():
            if grade_dir.is_dir() and grade_dir.name in self.quality_grades:
                quality = grade_dir.name
                file_count = len([f for f in grade_dir.iterdir() 
                                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
                quality_counts[quality] = file_count
                total_files += file_count
        
        # Create visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Quality distribution
        qualities = list(quality_counts.keys())
        counts = list(quality_counts.values())
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red']
        
        axes[0].bar(qualities, counts, color=colors)
        axes[0].set_title('Quality Grade Distribution')
        axes[0].set_xlabel('Quality Grade')
        axes[0].set_ylabel('Number of Samples')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[0].text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # Dataset summary
        axes[1].text(0.1, 0.8, f'Total Samples: {total_files}', fontsize=14, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.7, f'Quality Grades: {len(quality_counts)}', fontsize=14, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.6, f'Average per Grade: {total_files // len(quality_counts)}', fontsize=14, transform=axes[1].transAxes)
        
        # Add quality grade descriptions
        y_pos = 0.5
        for quality in qualities:
            axes[1].text(0.1, y_pos, f'{quality}: {quality_counts[quality]} samples', 
                        fontsize=12, transform=axes[1].transAxes)
            y_pos -= 0.08
        
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('Dataset Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig

