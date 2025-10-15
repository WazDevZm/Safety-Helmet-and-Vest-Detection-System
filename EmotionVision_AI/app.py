"""
EmotionVision AI - Flask Web Application
Real-time emotion detection with modern web interface
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import base64
import json
from emotion_detector import detector, init_detector, cleanup_detector
import threading
import time

app = Flask(__name__)

# Global variables for video streaming
current_frame = None
frame_lock = threading.Lock()

def generate_frames():
    """Generate video frames for streaming"""
    global current_frame
    
    while True:
        if detector.is_running:
            frame, emotions = detector.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                with frame_lock:
                    current_frame = frame_bytes
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/emotion')
def get_current_emotion():
    """Get current emotion data"""
    return jsonify({
        'emotion': detector.current_emotion,
        'confidence': detector.confidence,
        'timestamp': time.time()
    })

@app.route('/api/stats')
def get_emotion_stats():
    """Get emotion statistics"""
    stats = detector.get_emotion_stats()
    return jsonify(stats)

@app.route('/api/start')
def start_detection():
    """Start emotion detection"""
    if not detector.is_running:
        success = detector.start_camera()
        if success:
            return jsonify({'status': 'success', 'message': 'Detection started'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to start camera'})
    return jsonify({'status': 'already_running', 'message': 'Detection already running'})

@app.route('/api/stop')
def stop_detection():
    """Stop emotion detection"""
    detector.stop_camera()
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/api/reset')
def reset_stats():
    """Reset emotion statistics"""
    detector.emotion_history = []
    return jsonify({'status': 'success', 'message': 'Statistics reset'})

@app.route('/api/emotions/history')
def get_emotion_history():
    """Get emotion history"""
    return jsonify(detector.emotion_history[-20:])  # Last 20 emotions

if __name__ == '__main__':
    # Initialize detector
    init_detector()
    
    print("🎭 EmotionVision AI - Starting Web Application")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📱 Features:")
    print("   • Real-time emotion detection")
    print("   • Live video streaming")
    print("   • Emotion statistics")
    print("   • Modern web interface")
    print("   • Mobile responsive design")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down EmotionVision AI...")
    finally:
        cleanup_detector()

