"""
Flask Web Application for Face Anti-Spoofing System
Provides a modern web interface for the anti-spoofing detection
"""

from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import time
import os
from datetime import datetime
import logging
import numpy as np
import json
from threading import Lock

app = Flask(__name__)

# Setup logging
log_dir = "logs"
spoof_dir = "spoof_images"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(spoof_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"web_antispoofing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Global variables
camera = None
camera_lock = Lock()

# Tracking variables
blink_history = []
eye_closed_frames = 0
consecutive_no_face = 0
session_start = time.time()
last_status = None
spoof_save_cooldown = 0
prev_gray = None
motion_history = []

# Configuration
BLINK_THRESHOLD = 3
BLINK_WINDOW = 10
MIN_BLINKS_FOR_REAL = 2
NO_FACE_THRESHOLD = 30
SPOOF_SAVE_INTERVAL = 5
MOTION_THRESHOLD = 1000

# Stats for dashboard
stats = {
    'total_blinks': 0,
    'spoof_count': 0,
    'real_count': 0,
    'session_duration': 0,
    'current_status': 'WAITING',
    'recent_blinks': 0,
    'motion_level': 0,
    'face_detected': False
}

def get_camera():
    """Get camera instance (singleton)"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return None
    return camera

def calculate_motion(prev_frame, curr_frame):
    """Calculate motion between frames"""
    if prev_frame is None:
        return 0
    diff = cv2.absdiff(prev_frame, curr_frame)
    motion_score = np.sum(diff)
    return motion_score

def check_blink_validity():
    """Check if recent blinks indicate a real person"""
    current_time = time.time()
    valid_blinks = [t for t in blink_history if current_time - t < BLINK_WINDOW]
    return len(valid_blinks) >= MIN_BLINKS_FOR_REAL

def save_spoof_image(frame, reason):
    """Save potential spoof image with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = os.path.join(spoof_dir, f"spoof_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    logger.warning(f"Spoof detected - Reason: {reason} - Image saved: {filename}")
    return filename

def draw_info_panel(frame, status, face_detected):
    """Draw information panel on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Status
    color = (0, 255, 0) if status == "REAL PERSON" else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Blinks count
    recent_blinks = len([t for t in blink_history if time.time() - t < BLINK_WINDOW])
    cv2.putText(frame, f"Blinks: {recent_blinks}/{MIN_BLINKS_FOR_REAL}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Face detection indicator
    face_color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.circle(frame, (w-30, 30), 10, face_color, -1)

def generate_frames():
    """Generate video frames for streaming"""
    global eye_closed_frames, consecutive_no_face, last_status
    global spoof_save_cooldown, prev_gray, stats
    
    while True:
        with camera_lock:
            cap = get_camera()
            if cap is None:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
        
        current_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate motion
        motion_score = calculate_motion(prev_gray, gray)
        if prev_gray is not None:
            motion_history.append(motion_score)
            if len(motion_history) > 30:
                motion_history.pop(0)
        prev_gray = gray.copy()
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
        face_detected = len(faces) > 0
        stats['face_detected'] = face_detected
        
        if len(faces) == 0:
            consecutive_no_face += 1
            if consecutive_no_face > NO_FACE_THRESHOLD:
                if len(blink_history) > 0:
                    logger.info("Face lost - Resetting session")
                blink_history.clear()
                eye_closed_frames = 0
                consecutive_no_face = 0
        else:
            consecutive_no_face = 0
            
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Region of interest for eyes
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
                
                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Blink detection logic
                if len(eyes) < 2:
                    eye_closed_frames += 1
                else:
                    if eye_closed_frames >= BLINK_THRESHOLD:
                        blink_history.append(current_time)
                        stats['total_blinks'] += 1
                        logger.info(f"Blink detected! Total blinks: {stats['total_blinks']}")
                    eye_closed_frames = 0
        
        # Determine status
        is_real = check_blink_validity()
        avg_motion = np.mean(motion_history) if motion_history else 0
        stats['motion_level'] = int(avg_motion)
        
        recent_blinks = len([t for t in blink_history if current_time - t < BLINK_WINDOW])
        stats['recent_blinks'] = recent_blinks
        
        if is_real and avg_motion > MOTION_THRESHOLD:
            status = "REAL PERSON"
            if last_status != "REAL":
                logger.info("Status changed to REAL PERSON")
                last_status = "REAL"
                stats['real_count'] += 1
        else:
            status = "SPOOF DETECTED"
            
            if current_time - spoof_save_cooldown > SPOOF_SAVE_INTERVAL:
                reasons = []
                if not is_real:
                    reasons.append("Insufficient blinks")
                if avg_motion <= MOTION_THRESHOLD:
                    reasons.append("Low motion")
                
                reason = ", ".join(reasons)
                save_spoof_image(frame, reason)
                spoof_save_cooldown = current_time
                stats['spoof_count'] += 1
            
            if last_status != "SPOOF":
                logger.warning("Status changed to SPOOF DETECTED")
                last_status = "SPOOF"
        
        stats['current_status'] = status
        stats['session_duration'] = int(current_time - session_start)
        
        # Draw information panel
        draw_info_panel(frame, status, face_detected)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(stats)

@app.route('/reset')
def reset():
    """Reset the detection system"""
    global blink_history, eye_closed_frames, motion_history
    blink_history.clear()
    eye_closed_frames = 0
    motion_history.clear()
    logger.info("Manual reset via web interface")
    return jsonify({'status': 'success', 'message': 'System reset'})

@app.route('/logs')
def view_logs():
    """View recent logs"""
    try:
        with open(log_filename, 'r') as f:
            lines = f.readlines()
            recent_logs = lines[-50:]  # Last 50 lines
        return jsonify({'logs': recent_logs})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/spoof_images')
def list_spoof_images():
    """List saved spoof images"""
    try:
        images = [f for f in os.listdir(spoof_dir) if f.endswith('.jpg')]
        images.sort(reverse=True)  # Most recent first
        return jsonify({'images': images[:10]})  # Return last 10
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/spoof_images/<filename>')
def get_spoof_image(filename):
    """Serve spoof image"""
    return send_from_directory(spoof_dir, filename)

if __name__ == '__main__':
    logger.info("Face Anti-Spoofing Web Application Started")
    print("\n" + "="*70)
    print("🚀 Face Anti-Spoofing System - Web Interface")
    print("="*70)
    print(f"📱 Open your browser and go to: http://localhost:5000")
    print(f"📊 View statistics at: http://localhost:5000/stats")
    print(f"🔄 Reset system at: http://localhost:5000/reset")
    print("="*70 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)