"""
Ore Quality Classification System
CNN-based classification of ore samples using image recognition
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class OreClassifier:
    """
    CNN-based Ore Quality Classification System
    Classifies ore samples into quality grades based on visual characteristics
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        """
        Initialize the ore classifier
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of ore quality classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            'Very High Grade',
            'High Grade', 
            'Medium Grade',
            'Low Grade',
            'Very Low Grade'
        ]
        self.class_colors = {
            'Very High Grade': (0, 255, 0),      # Green
            'High Grade': (0, 200, 0),           # Light Green
            'Medium Grade': (255, 255, 0),       # Yellow
            'Low Grade': (255, 165, 0),          # Orange
            'Very Low Grade': (255, 0, 0)        # Red
        }
        
    def create_model(self):
        """
        Create CNN model for ore classification
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            # Dense layers
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for classification
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract visual features from ore image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Dictionary of extracted features
        """
        # Remove batch dimension for feature extraction
        img = image[0] if len(image.shape) == 4 else image
        
        # Convert to uint8 for OpenCV operations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Color analysis
        mean_color = np.mean(img_uint8, axis=(0, 1))
        std_color = np.std(img_uint8, axis=(0, 1))
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        mean_hsv = np.mean(hsv, axis=(0, 1))
        
        # Texture analysis using LBP (Local Binary Pattern)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features
        texture_features = self._calculate_texture_features(gray)
        
        # Surface characteristics
        surface_features = self._calculate_surface_features(img_uint8)
        
        features = {
            'mean_red': float(mean_color[0]),
            'mean_green': float(mean_color[1]),
            'mean_blue': float(mean_color[2]),
            'std_red': float(std_color[0]),
            'std_green': float(std_color[1]),
            'std_blue': float(std_color[2]),
            'mean_hue': float(mean_hsv[0]),
            'mean_saturation': float(mean_hsv[1]),
            'mean_value': float(mean_hsv[2]),
            **texture_features,
            **surface_features
        }
        
        return features
    
    def _calculate_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Calculate texture features using various methods"""
        features = {}
        
        # GLCM (Gray-Level Co-occurrence Matrix) features
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Calculate GLCM
            glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            
            # Extract texture properties
            features['contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
            features['dissimilarity'] = float(graycoprops(glcm, 'dissimilarity')[0, 0])
            features['homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
            features['energy'] = float(graycoprops(glcm, 'energy')[0, 0])
            features['correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
        except ImportError:
            # Fallback texture features
            features['contrast'] = float(np.std(gray_image))
            features['dissimilarity'] = float(np.mean(np.abs(np.diff(gray_image.flatten()))))
            features['homogeneity'] = float(1.0 / (1.0 + np.var(gray_image)))
            features['energy'] = float(np.sum(gray_image**2) / (gray_image.shape[0] * gray_image.shape[1]))
            features['correlation'] = 0.0
        
        # Additional texture features
        features['texture_std'] = float(np.std(gray_image))
        features['texture_mean'] = float(np.mean(gray_image))
        
        return features
    
    def _calculate_surface_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate surface characteristics"""
        features = {}
        
        # Edge detection for surface roughness
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Surface roughness indicators
        features['edge_density'] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
        
        # Brightness and contrast
        features['brightness'] = float(np.mean(gray))
        features['contrast'] = float(np.std(gray))
        
        # Color distribution
        features['color_variance'] = float(np.var(image))
        
        return features
    
    def predict_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Predict ore quality from image
        
        Args:
            image_path: Path to ore image
            
        Returns:
            Prediction results with confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Extract features
        features = self.extract_features(processed_image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': self.class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        return {
            'predicted_class': self.class_names[predicted_class_idx],
            'confidence': confidence,
            'class_index': int(predicted_class_idx),
            'top_3_predictions': top_3_predictions,
            'features': features,
            'all_probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, predictions[0])
            }
        }
    
    def train_model(self, data_dir: str, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
        """
        Train the ore classification model
        
        Args:
            data_dir: Directory containing training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Load and prepare data
        train_generator, val_generator = self._prepare_data_generators(
            data_dir, batch_size, validation_split
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_ore_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _prepare_data_generators(self, data_dir: str, batch_size: int, validation_split: float):
        """Prepare data generators for training"""
        # Data augmentation for training
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            validation_split=validation_split
        )
        
        # Validation data generator (no augmentation)
        val_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        val_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def evaluate_model(self, test_data_dir: str) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data_dir: Directory containing test data
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Prepare test data
        test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate model
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Generate classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'top_3_accuracy': results[2],
            'classification_report': report,
            'confusion_matrix': confusion_matrix(true_classes, predicted_classes)
        }
    
    def visualize_predictions(self, image_path: str, save_path: str = None):
        """
        Visualize prediction results
        
        Args:
            image_path: Path to ore image
            save_path: Path to save visualization
        """
        # Get prediction
        result = self.predict_quality(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Ore Sample')
        axes[0].axis('off')
        
        # Prediction results
        classes = list(result['all_probabilities'].keys())
        probabilities = list(result['all_probabilities'].values())
        colors = [self.class_colors[cls] for cls in classes]
        
        bars = axes[1].barh(classes, probabilities, color=[c/255 for c in colors])
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_title(f'Predicted: {result["predicted_class"]} ({result["confidence"]:.2f})')
        axes[1].set_xlim(0, 1)
        
        # Add confidence values on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            axes[1].text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig

