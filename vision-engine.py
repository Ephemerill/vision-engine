import warnings
warnings.filterwarnings("ignore")

import cv2
import face_recognition 
import os
import numpy as np
import threading
import time
import logging
import requests
from ultralytics import YOLO
from flask import Flask, Response

# --- CONFIGURATION ---
# TAILSCALE IP
STREAM_URL = "rtsp://admin:mysecretpassword@100.114.210.58:8554/cam"
WEB_SERVER_PORT = 5005

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisionEngine")

# --- FLASK ---
app = Flask(__name__)
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

# --- GLOBALS ---
lock = threading.Lock()
# The "Hot" Frame - We only keep ONE.
latest_frame = None
output_frame = None

# --- MODELS ---
logger.info("Loading Models...")
yolo_body = YOLO("yolo11n.pt", verbose=False)
yolo_face = YOLO("yolov11n-face.pt", verbose=False) # Ensure you have this file
known_face_encodings = []
known_face_names = []

# --- LOAD FACES ---
if os.path.exists("known_faces"):
    for name in os.listdir("known_faces"):
        dir_path = os.path.join("known_faces", name)
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(('.jpg', '.png')):
                    try:
                        img = face_recognition.load_image_file(os.path.join(dir_path, f))
                        enc = face_recognition.face_encodings(img)[0]
                        known_face_encodings.append(enc)
                        known_face_names.append(name)
                    except: pass

# --- THREAD 1: THE SIPHON (Reads Stream, Dumps Buffer) ---
def capture_thread():
    global latest_frame
    logger.info(f"Connecting to: {STREAM_URL}")
    
    # FFMPEG BACKEND WITH NO BUFFER
    # This forces OpenCV to ask FFMPEG to drop frames if they are old
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        logger.error("Failed to connect to RTSP stream.")
        return

    logger.info("Stream Connected. Siphoning frames...")
    
    while True:
        # Grab frame (blocking)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame dropped, reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
            continue
            
        # ATOMIC WRITE
        # We don't care if we overwrite data the processor hasn't seen.
        # We only want the processor to see the NEWEST data.
        with lock:
            latest_frame = frame

# --- THREAD 2: THE PROCESSOR (Runs AI on 'latest_frame') ---
def process_thread():
    global output_frame, latest_frame
    
    last_processed_time = 0
    face_locs = []
    face_names = []
    
    while True:
        # 1. GET LATEST
        frame_to_process = None
        with lock:
            if latest_frame is not None:
                frame_to_process = latest_frame.copy()
        
        if frame_to_process is None:
            time.sleep(0.01)
            continue
            
        # 2. RESIZE (Critical for FPS)
        frame_small = cv2.resize(frame_to_process, (640, 480))
        
        # 3. RUN AI (Every 3rd frame effectively, or based on time)
        # We run detection on every frame here because the thread is decoupled
        # If AI is slow, it just skips the frames the Capture Thread captured in the meantime.
        
        # BODY
        results = yolo_body.track(frame_small, persist=True, verbose=False, classes=[0]) # 0=Person
        
        # FACE (Simple Logic)
        if time.time() - last_processed_time > 0.2: # Run face rec 5 times a second max
            last_processed_time = time.time()
            # Detect
            face_results = yolo_face.predict(frame_small, verbose=False, conf=0.5)
            face_locs = []
            if len(face_results) > 0:
                 for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    face_locs.append((t, r, b, l))
            
            # Recognize
            rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb, face_locs)
            face_names = []
            for enc in encodings:
                matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    name = known_face_names[matches.index(True)]
                face_names.append(name)

        # 4. DRAW
        # Draw Bodies
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, id in zip(boxes, ids):
                l, t, r, b = box
                cv2.rectangle(frame_small, (l, t), (r, b), (255, 100, 100), 2)
                cv2.putText(frame_small, f"ID: {id}", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Draw Faces
        for (t, r, b, l), name in zip(face_locs, face_names):
            cv2.rectangle(frame_small, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame_small, name, (l, b+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 5. UPDATE OUTPUT
        with lock:
            output_frame = frame_small

# --- FLASK SERVER ---
def generate():
    while True:
        with lock:
            if output_frame is None: 
                time.sleep(0.01)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- RUN ---
if __name__ == "__main__":
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=process_thread, daemon=True)
    t1.start()
    t2.start()
    
    print(f"Server starting on port {WEB_SERVER_PORT}...")
    app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)