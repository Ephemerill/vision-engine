import warnings
warnings.filterwarnings("ignore")

import cv2
import face_recognition 
import os
import numpy as np
import threading
import time
import logging
import sys
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# --- CONFIGURATION ---
# UPDATE THIS IP to your Pi's Tailscale IP
STREAM_PI_IP = "100.114.210.58" 
STREAM_URL = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam"
WEB_SERVER_PORT = 5005

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisionEngine")

# --- FLASK APP ---
app = Flask(__name__)
# Silence Flask logs to keep console clean
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# --- GLOBAL STATE ---
lock = threading.Lock()
latest_frame = None   # The raw frame from the camera
output_frame = None   # The annotated frame for the browser
is_connected = False  # Track connection status

# --- MODELS ---
print("------------------------------------------------")
logger.info("Loading AI Models (this may take a moment)...")
try:
    yolo_body = YOLO("yolo11n.pt", verbose=False)
    yolo_face = YOLO("yolov11n-face.pt", verbose=False)
except Exception as e:
    logger.error(f"Failed to load models: {e}")
    sys.exit(1)

# --- LOAD KNOWN FACES ---
known_face_encodings = []
known_face_names = []
if os.path.exists("known_faces"):
    logger.info("Loading Known Faces...")
    for name in os.listdir("known_faces"):
        dir_path = os.path.join("known_faces", name)
        if os.path.isdir(dir_path):
            for f in os.listdir(dir_path):
                if f.endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        img = face_recognition.load_image_file(os.path.join(dir_path, f))
                        encs = face_recognition.face_encodings(img)
                        if encs:
                            known_face_encodings.append(encs[0])
                            known_face_names.append(name)
                    except: pass
    logger.info(f"Loaded {len(known_face_names)} faces.")

# --- HELPER: CREATE BLANK FRAME ---
def create_placeholder(text="Waiting for stream..."):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

# Initialize output with a placeholder
output_frame = create_placeholder()

# --- THREAD 1: STREAM CAPTURE (The "Siphon") ---
def capture_thread():
    global latest_frame, is_connected
    
    # Force Low Latency flags for FFMPEG
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    while True:
        logger.info(f"Attempting connection to: {STREAM_URL}")
        cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            logger.error("❌ Connection Refused. Check if Pi Stream is running.")
            is_connected = False
            time.sleep(2)
            continue
            
        logger.info("✅ Stream Connected! Receiving frames...")
        is_connected = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Frame dropped or stream ended. Reconnecting...")
                is_connected = False
                break
            
            # ATOMIC UPDATE: Just overwrite the variable.
            # We don't wait for locks here to ensure max read speed.
            latest_frame = frame
            
        cap.release()
        time.sleep(1)

# --- THREAD 2: AI PROCESSOR ---
def process_thread():
    global output_frame, latest_frame
    
    last_processed_time = 0
    face_locs = []
    face_names = []
    
    logger.info("AI Processor Started.")
    
    while True:
        # 1. GRAB LATEST FRAME
        # If no frame is available yet, just update the placeholder status
        if latest_frame is None:
            if not is_connected:
                with lock: output_frame = create_placeholder("Connecting to Pi...")
            else:
                with lock: output_frame = create_placeholder("Buffering...")
            time.sleep(0.1)
            continue
            
        # Copy frame to avoid tearing/conflicts
        frame = latest_frame.copy()
        
        # 2. RESIZE (Crucial for performance)
        # We force it to 640x480 for consistent processing speed
        frame_small = cv2.resize(frame, (640, 480))
        
        # 3. RUN AI
        # Body Tracking
        try:
            results = yolo_body.track(frame_small, persist=True, verbose=False, classes=[0], conf=0.4)
            
            # Face Detection (Throttled to 5 FPS to save CPU)
            if time.time() - last_processed_time > 0.2:
                last_processed_time = time.time()
                
                face_results = yolo_face.predict(frame_small, verbose=False, conf=0.5)
                face_locs = []
                if len(face_results) > 0 and face_results[0].boxes is not None:
                     for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                        l, t, r, b = box
                        face_locs.append((t, r, b, l))
                
                # Recognition
                if face_locs:
                    rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb, face_locs)
                    face_names = []
                    for enc in encodings:
                        matches = face_recognition.compare_faces(known_face_encodings, enc, tolerance=0.5)
                        name = "Unknown"
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]
                        face_names.append(name)
                else:
                    face_names = []

            # 4. DRAWING
            # Draw Bodies
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    l, t, r, b = box
                    cv2.rectangle(frame_small, (l, t), (r, b), (255, 100, 100), 2)
                    cv2.putText(frame_small, f"ID: {id}", (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Draw Faces
            if len(face_locs) == len(face_names):
                for (t, r, b, l), name in zip(face_locs, face_names):
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame_small, (l, t), (r, b), color, 2)
                    cv2.putText(frame_small, name, (l, b+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        except Exception as e:
            # If AI fails (e.g. corruption), just skip this frame
            pass

        # 5. UPDATE OUTPUT
        with lock:
            output_frame = frame_small

# --- FLASK ROUTES ---
@app.route('/')
def index():
    # THE MISSING PIECE: A simple HTML page to view the stream
    return """
    <html>
        <head>
            <title>Vision Engine</title>
            <style>
                body { background-color: #111; color: white; text-align: center; font-family: sans-serif; margin-top: 50px; }
                img { border: 2px solid #444; border-radius: 8px; box-shadow: 0 0 20px rgba(0,0,0,0.5); max-width: 100%; }
                h1 { margin-bottom: 10px; }
                p { color: #888; }
            </style>
        </head>
        <body>
            <h1>Vision Engine Live</h1>
            <p>Low Latency Receiver</p>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    """

def generate():
    while True:
        with lock:
            if output_frame is None:
                frame = create_placeholder()
            else:
                frame = output_frame
            
            (flag, encodedImage) = cv2.imencode(".jpg", frame)
            if not flag: continue
            
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        # Cap the browser stream at 30fps to save bandwidth
        time.sleep(0.03) 

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- MAIN ---
if __name__ == "__main__":
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=process_thread, daemon=True)
    t1.start()
    t2.start()
    
    print("------------------------------------------------")
    print(f"✅ WEB SERVER READY: http://0.0.0.0:{WEB_SERVER_PORT}")
    print("------------------------------------------------")
    
    app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)