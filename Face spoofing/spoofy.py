"""
Cloud-Compatible Face Anti-Spoofing System
Works on Render, Heroku, etc. because camera runs in browser
"""

from flask import Flask, render_template, jsonify, request
import cv2
import time
import os
from datetime import datetime
import logging
import numpy as np
import base64

app = Flask(__name__)

# Setup logging
log_dir = "logs"
spoof_dir = "spoof_images"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(spoof_dir, exist_ok=True)

log_filename = os.path.join(log_dir, f"cloud_antispoofing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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

# Session storage (in production, use Redis or database)
sessions = {}

# Configuration
BLINK_THRESHOLD = 3
BLINK_WINDOW = 10
MIN_BLINKS_FOR_REAL = 2

def get_session(session_id):
    """Get or create session data"""
    if session_id not in sessions:
        sessions[session_id] = {
            'blink_history': [],
            'eye_closed_frames': 0,
            'stats': {
                'total_blinks': 0,
                'spoof_count': 0,
                'real_count': 0,
                'session_duration': 0,
                'current_status': 'WAITING',
                'recent_blinks': 0,
                'face_detected': False
            },
            'start_time': time.time()
        }
    return sessions[session_id]

def check_blink_validity(blink_history):
    """Check if recent blinks indicate a real person"""
    current_time = time.time()
    valid_blinks = [t for t in blink_history if current_time - t < BLINK_WINDOW]
    return len(valid_blinks) >= MIN_BLINKS_FOR_REAL

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frame sent from browser"""
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        image_data = data['image']
        
        # Get session
        session = get_session(session_id)
        
        # Decode image
        encoded_data = image_data.split(',')[1]
        decoded_data = base64.b64decode(encoded_data)
        np_arr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        face_detected = len(faces) > 0
        session['stats']['face_detected'] = face_detected
        
        current_time = time.time()
        
        # Process face detection
        if face_detected:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(15, 15))
                
                # Blink detection
                if len(eyes) < 2:
                    session['eye_closed_frames'] += 1
                else:
                    if session['eye_closed_frames'] >= BLINK_THRESHOLD:
                        session['blink_history'].append(current_time)
                        session['stats']['total_blinks'] += 1
                        logger.info(f"Blink detected - Session: {session_id}")
                    session['eye_closed_frames'] = 0
        
        # Determine status
        is_real = check_blink_validity(session['blink_history'])
        recent_blinks = len([t for t in session['blink_history'] if current_time - t < BLINK_WINDOW])
        session['stats']['recent_blinks'] = recent_blinks
        
        # Update status
        prev_status = session['stats']['current_status']
        if is_real:
            session['stats']['current_status'] = "REAL PERSON"
            if prev_status != "REAL PERSON":
                session['stats']['real_count'] += 1
        else:
            session['stats']['current_status'] = "SPOOF DETECTED"
            if prev_status != "SPOOF DETECTED" and prev_status != "WAITING":
                session['stats']['spoof_count'] += 1
                # Save spoof image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(spoof_dir, f"spoof_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                logger.warning(f"Spoof detected - Image saved: {filename}")
        
        session['stats']['session_duration'] = int(current_time - session['start_time'])
        
        return jsonify({
            'status': session['stats']['current_status'],
            'face_detected': face_detected,
            'total_blinks': session['stats']['total_blinks'],
            'recent_blinks': session['stats']['recent_blinks'],
            'real_count': session['stats']['real_count'],
            'spoof_count': session['stats']['spoof_count'],
            'session_duration': session['stats']['session_duration']
        })
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    session_id = request.args.get('session_id', 'default')
    session = get_session(session_id)
    return jsonify(session['stats'])

@app.route('/reset')
def reset():
    """Reset the detection system"""
    session_id = request.args.get('session_id', 'default')
    if session_id in sessions:
        del sessions[session_id]
    logger.info(f"Session reset: {session_id}")
    return jsonify({'status': 'success', 'message': 'System reset'})

@app.route('/logs')
def view_logs():
    """View recent logs"""
    try:
        with open(log_filename, 'r') as f:
            lines = f.readlines()
            recent_logs = lines[-50:]
        return jsonify({'logs': recent_logs})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check for cloud platforms"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(sessions),
        'platform': 'cloud'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info("Cloud Face Anti-Spoofing System Started")
    app.run(host='0.0.0.0', port=port)

