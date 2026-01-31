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
os.makedirs(log_dir, exist_ok=True)

log_filename = os.path.join(
    log_dir,
    f"web_antispoofing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Detection variables
blink_history = []
eye_closed_frames = 0
session_start = time.time()

BLINK_THRESHOLD = 3
BLINK_WINDOW = 10
MIN_BLINKS_FOR_REAL = 2

stats = {
    'total_blinks': 0,
    'real_count': 0,
    'spoof_count': 0,
    'recent_blinks': 0,
    'current_status': 'WAITING',
    'session_duration': 0,
    'face_detected': False
}


def check_blink_validity():
    current_time = time.time()
    valid_blinks = [t for t in blink_history if current_time - t < BLINK_WINDOW]
    return len(valid_blinks) >= MIN_BLINKS_FOR_REAL


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global eye_closed_frames, blink_history, stats

    data = request.json['image']
    encoded_data = data.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)

    np_arr = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_detected = len(faces) > 0
    stats['face_detected'] = face_detected

    current_time = time.time()

    if face_detected:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)

            if len(eyes) < 2:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= BLINK_THRESHOLD:
                    blink_history.append(current_time)
                    stats['total_blinks'] += 1
                    logger.info("Blink detected")
                eye_closed_frames = 0

    is_real = check_blink_validity()

    recent_blinks = len([t for t in blink_history if current_time - t < BLINK_WINDOW])
    stats['recent_blinks'] = recent_blinks

    if is_real:
        stats['current_status'] = "REAL PERSON"
        stats['real_count'] += 1
    else:
        stats['current_status'] = "SPOOF DETECTED"
        stats['spoof_count'] += 1

    stats['session_duration'] = int(current_time - session_start)

    return jsonify({
        "status": stats['current_status'],
        "face_detected": face_detected,
        "total_blinks": stats['total_blinks'],
        "recent_blinks": stats['recent_blinks']
    })


@app.route('/stats')
def get_stats():
    return jsonify(stats)


@app.route('/reset')
def reset():
    global blink_history, eye_closed_frames, stats

    blink_history.clear()
    eye_closed_frames = 0

    stats['total_blinks'] = 0
    stats['recent_blinks'] = 0
    stats['real_count'] = 0
    stats['spoof_count'] = 0
    stats['current_status'] = "WAITING"

    return jsonify({'status': 'success', 'message': 'System reset'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
