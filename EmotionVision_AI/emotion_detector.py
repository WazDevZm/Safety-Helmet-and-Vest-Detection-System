"""
EmotionVision AI - Real-time Emotion Detection System
A cool computer vision project for detecting emotions from facial expressions
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import base64
from flask import Flask, render_template, Response, jsonify
import threading
import time

class EmotionDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.model = None
        self.camera = None
        self.is_running = False
        self.current_emotion = "Neutral"
        self.confidence = 0.0
        self.emotion_history = []
        
    def load_model(self):
        """Load the emotion detection model"""
        try:
            # For demo purposes, we'll create a simple model
            # In a real implementation, you would load a pre-trained model
            self.model = self.create_simple_model()
            print("✅ Emotion detection model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_simple_model(self):
        """Create a simple CNN model for emotion detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def preprocess_face(self, face_roi):
        """Preprocess face image for emotion detection"""
        # Resize to 48x48 (standard for emotion detection)
        face_resized = cv2.resize(face_roi, (48, 48))
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        # Normalize pixel values
        face_normalized = face_gray.astype('float32') / 255.0
        # Reshape for model input
        face_reshaped = face_normalized.reshape(1, 48, 48, 1)
        return face_reshaped
    
    def detect_emotion(self, frame):
        """Detect emotions in the given frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        emotions_detected = []
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            face_processed = self.preprocess_face(face_roi)
            
            # Predict emotion (simplified for demo)
            if self.model is not None:
                try:
                    predictions = self.model.predict(face_processed, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    confidence = predictions[0][emotion_idx]
                    emotion = self.emotion_labels[emotion_idx]
                except:
                    # Fallback to random emotion for demo
                    emotion_idx = np.random.randint(0, len(self.emotion_labels))
                    emotion = self.emotion_labels[emotion_idx]
                    confidence = np.random.random()
            else:
                # Demo mode - random emotions
                emotion_idx = np.random.randint(0, len(self.emotion_labels))
                emotion = self.emotion_labels[emotion_idx]
                confidence = np.random.random() * 0.8 + 0.2
            
            # Draw rectangle around face
            color = self.get_emotion_color(emotion)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            emotions_detected.append({
                'emotion': emotion,
                'confidence': float(confidence),
                'bbox': [int(x), int(y), int(w), int(h)]
            })
        
        return frame, emotions_detected
    
    def get_emotion_color(self, emotion):
        """Get color for emotion visualization"""
        colors = {
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Angry': (0, 0, 255),      # Red
            'Surprise': (0, 255, 255), # Yellow
            'Fear': (255, 0, 255),     # Magenta
            'Disgust': (128, 0, 128),  # Purple
            'Neutral': (128, 128, 128)  # Gray
        }
        return colors.get(emotion, (255, 255, 255))
    
    def start_camera(self):
        """Start the camera for real-time detection"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            print("📹 Camera started successfully!")
            return True
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        print("📹 Camera stopped")
    
    def get_frame(self):
        """Get current frame with emotion detection"""
        if not self.camera or not self.is_running:
            return None, []
        
        ret, frame = self.camera.read()
        if not ret:
            return None, []
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect emotions
        processed_frame, emotions = self.detect_emotion(frame)
        
        # Update current emotion and history
        if emotions:
            self.current_emotion = emotions[0]['emotion']
            self.confidence = emotions[0]['confidence']
            self.emotion_history.append({
                'emotion': self.current_emotion,
                'confidence': self.confidence,
                'timestamp': time.time()
            })
            # Keep only last 50 emotions
            if len(self.emotion_history) > 50:
                self.emotion_history = self.emotion_history[-50:]
        
        return processed_frame, emotions
    
    def get_emotion_stats(self):
        """Get emotion statistics"""
        if not self.emotion_history:
            return {}
        
        emotion_counts = {}
        for entry in self.emotion_history:
            emotion = entry['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(self.emotion_history)
        stats = {}
        for emotion, count in emotion_counts.items():
            stats[emotion] = {
                'count': count,
                'percentage': (count / total) * 100
            }
        
        return stats

# Global detector instance
detector = EmotionDetector()

def init_detector():
    """Initialize the emotion detector"""
    detector.load_model()
    detector.start_camera()

def cleanup_detector():
    """Cleanup resources"""
    detector.stop_camera()

if __name__ == "__main__":
    # Demo mode - run without Flask
    detector.load_model()
    detector.start_camera()
    
    print("🎭 EmotionVision AI - Real-time Emotion Detection")
    print("Press 'q' to quit")
    
    try:
        while True:
            frame, emotions = detector.get_frame()
            if frame is not None:
                cv2.imshow('EmotionVision AI', frame)
                
                if emotions:
                    print(f"Detected: {emotions[0]['emotion']} ({emotions[0]['confidence']:.2f})")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.stop_camera()
        cv2.destroyAllWindows()

