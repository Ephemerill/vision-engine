import warnings
# --- SILENCE WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import face_recognition 
import os
import numpy as np
import threading
import time
import sys
import socket
import logging
import requests
import queue
import datetime
import torch
from flask import Flask, Response, make_response
from ultralytics import YOLO

# --- CONFIGURATION ---
RECOGNITION_TOLERANCE = 0.5 
WEB_SERVER_PORT = 5005
DIAGNOSTICS_INTERVAL = 5.0 
HIGH_CONFIDENCE_THRESHOLD = 65.0 
RECOGNITION_COOLDOWN = 2.0 
UNKNOWN_RETRY_LIMIT = 5          
UNKNOWN_LOCKOUT_TIME = 30.0      

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisionEngine")
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

# --- HARDWARE CHECK ---
if torch.cuda.is_available():
    DEVICE_STR = 'cuda:0'
else:
    DEVICE_STR = 'cpu'

# --- FLASK APP ---
app = Flask(__name__)

# --- SETTINGS ---
STREAM_PI_IP = "100.114.210.58"
# Aggressive RTSP flags to kill latency at the source
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp&buffer_size=0"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 
FACE_RECOGNITION_NTH_FRAME = 3 

# --- GLOBAL STATE ---
results_lock = threading.Lock()
registry_lock = threading.Lock()
output_lock = threading.Lock()

# This variable holds the ONE latest JPEG image. 
# We overwrite it constantly. We never queue it.
latest_jpeg_bytes = None

latest_processed_results = {  
    "body_boxes": {},
    "active_ids": [],
    "face_boxes": [],
}

person_registry = {} 
recog_queue = queue.Queue(maxsize=1) 
APP_SHOULD_QUIT = False

# --- MODELS ---
FACE_MODEL_NAME = "yolov11n-face.pt"
FACE_MODEL_URL = f"https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/{FACE_MODEL_NAME}"
yolo_body_model = None
yolo_face_model = None
known_face_encodings = []
known_face_names = []

# --- HELPER FUNCTIONS ---
def get_face_model_path():
    if os.path.exists(FACE_MODEL_NAME): return FACE_MODEL_NAME
    logger.info(f"Downloading {FACE_MODEL_NAME}...")
    try:
        r = requests.get(FACE_MODEL_URL, stream=True)
        with open(FACE_MODEL_NAME, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        return FACE_MODEL_NAME
    except: return None

def resize_with_aspect_ratio(frame, max_w=640, max_h=480):
    if frame is None: return None
    h, w = frame.shape[:2]
    if w == 0 or h == 0: return frame
    if w <= max_w and h <= max_h: return frame
    r = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names
    if not os.path.exists(known_faces_dir): return
    known_face_encodings.clear(); known_face_names.clear()
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                try:
                    img = face_recognition.load_image_file(os.path.join(person_dir, filename))
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        known_face_encodings.append(encs[0])
                        known_face_names.append(person_name)
                except: pass

def get_ip_addresses():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80))
        return [s.getsockname()[0]]
    except: return []

# --- AGGRESSIVE VIDEO READER ---
class FastVideoReader:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.connected = False
        
    def start(self):
        self.thread.start()
        return self

    def _run(self):
        # The "Nuclear Option" of FFmpeg flags to destroy latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|strict;experimental|analyzeduration;0|probesize;32"
        
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        
        while self.running:
            if not cap.isOpened():
                self.connected = False
                time.sleep(1)
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                continue
            
            # --- BUFFER FLUSH ---
            # We want the LATEST frame. If the buffer has old frames, 
            # we read() repeatedly until we hit a frame that makes us wait (live data).
            reads = 0
            while True:
                t1 = time.time()
                ret, frame = cap.read()
                t2 = time.time()
                reads += 1
                
                if not ret:
                    self.connected = False
                    cap.release()
                    break
                
                # If read() took > 5ms, it means we waited for the camera. This is a FRESH frame.
                # If read() was instant, it was a buffered frame. Discard it and loop again.
                if (t2 - t1) > 0.005 or reads > 5:
                    with self.lock:
                        self.frame = frame
                    self.connected = True
                    break

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        self.running = False
        self.thread.join()

stream_reader = None

# --- FLASK ROUTES ---

# 1. The HTML Page (Now with JavaScript Polling)
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Vision Engine Zero-Lat</title>
        <style>
            body { background: #111; color: white; text-align: center; margin: 0; padding: 20px; font-family: sans-serif; }
            #video-container { position: relative; display: inline-block; border: 2px solid #333; }
            #cam-feed { width: 100%; max-width: 800px; height: auto; display: block; }
            #status { margin-top: 10px; color: #0f0; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>Vision Engine | Zero Latency</h1>
        <div id="video-container">
            <img id="cam-feed" src="" alt="Connecting..." />
        </div>
        <div id="status">Connecting...</div>

        <script>
            const img = document.getElementById('cam-feed');
            const status = document.getElementById('status');
            
            // This function asks the server for ONE frame.
            // Only when the frame arrives does it ask for the next one.
            // This makes queueing/buffering impossible.
            function fetchFrame() {
                const startTime = Date.now();
                
                // Add timestamp to prevent browser caching
                fetch('/snapshot?t=' + startTime)
                    .then(response => response.blob())
                    .then(blob => {
                        const url = URL.createObjectURL(blob);
                        img.onload = () => {
                            URL.revokeObjectURL(url); // Clean up memory
                            
                            // Calculate FPS roughly
                            const latency = Date.now() - startTime;
                            status.innerText = 'Live | Latency: ' + latency + 'ms';
                            
                            // Immediately fetch the next one
                            requestAnimationFrame(fetchFrame);
                        }
                        img.src = url;
                    })
                    .catch(err => {
                        status.innerText = 'Connection Lost - Retrying...';
                        status.style.color = 'red';
                        setTimeout(fetchFrame, 1000);
                    });
            }

            // Start the loop
            fetchFrame();
        </script>
    </body>
    </html>
    """

# 2. The Snapshot Endpoint (Returns Single JPEG)
@app.route('/snapshot')
def snapshot():
    global latest_jpeg_bytes
    
    # If we have no frame yet, return a blank black image
    if latest_jpeg_bytes is None:
        blank = np.zeros((480, 640, 3), np.uint8)
        _, b = cv2.imencode(".jpg", blank)
        latest_jpeg_bytes = b.tobytes()

    response = make_response(latest_jpeg_bytes)
    response.headers['Content-Type'] = 'image/jpeg'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

def run_flask():
    try:
        # Threaded=True is important for handling multiple fetches quickly
        app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Flask Error: {e}")

# --- THREAD 2: RECOGNITION WORKER ---
def recognition_worker_thread():
    global person_registry
    logger.info("Background Recognition Worker Started.")
    
    while not APP_SHOULD_QUIT:
        try:
            job = recog_queue.get(timeout=0.1) 
        except queue.Empty: continue

        track_id = job["id"]
        rgb_frame = job["frame"]
        face_loc = job["loc"]
        
        try:
            encs = face_recognition.face_encodings(rgb_frame, [face_loc])
        except: encs = []

        new_name = "Unknown"; new_conf = 0.0
        if encs and len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, encs[0], tolerance=RECOGNITION_TOLERANCE)
            dists = face_recognition.face_distance(known_face_encodings, encs[0])
            if True in matches:
                best_idx = np.argmin(dists)
                new_name = known_face_names[best_idx]
                new_conf = max(0, min(100, (1.0 - dists[best_idx]) * 100))
        
        with registry_lock:
            if track_id in person_registry:
                entry = person_registry[track_id]
                if new_name != "Unknown":
                    entry["name"] = new_name; entry["conf"] = new_conf; entry["status"] = "known"; entry["retries"] = 0
                else:
                    entry["retries"] += 1
                    if entry["retries"] >= UNKNOWN_RETRY_LIMIT: entry["status"] = "visitor"
                entry["last_recog"] = time.time()

# --- THREAD 3: PROCESSING & DRAWING LOOP ---
def video_processing_thread():
    global latest_processed_results, person_registry, latest_jpeg_bytes
    frame_count = 0
    
    while stream_reader is None: time.sleep(0.5)

    logger.info("Processing Loop Started.")
    
    while not APP_SHOULD_QUIT:
        frame = stream_reader.read()
        if frame is None: time.sleep(0.005); continue
        
        # --- PROCESSING (YOLO) ---
        frame_sm = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        frame_count += 1
        
        # YOLO Body
        body_results = yolo_body_model.track(frame_sm, persist=True, classes=[0], conf=0.4, verbose=False)
        
        body_boxes = {}
        active_ids = []
        
        if body_results[0].boxes.id is not None:
            boxes = body_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = body_results[0].boxes.id.cpu().numpy().astype(int)
            
            with registry_lock:
                for box, track_id in zip(boxes, track_ids):
                    l, t, r, b = box
                    body_boxes[track_id] = (t, r, b, l)
                    active_ids.append(track_id)
                    if track_id not in person_registry:
                        person_registry[track_id] = {"name": "Unknown", "conf": 0.0, "last_recog": 0, "retries": 0, "status": "tracking"}

        # YOLO Face
        curr_faces = []
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model:
            face_results = yolo_face_model.predict(frame_sm, conf=FACE_CONFIDENCE_THRESH, verbose=False)
            if len(face_results) > 0:
                for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    curr_faces.append((t, r, b, l))
            
            # Recog Logic
            rgb = cv2.cvtColor(frame_sm, cv2.COLOR_BGR2RGB)
            for face_loc in curr_faces:
                # Basic overlap matching
                for bid, bbox in body_boxes.items():
                    # Check if face center is roughly inside body horizontal bounds
                    face_center_x = (face_loc[1] + face_loc[3]) / 2
                    if bbox[3] < face_center_x < bbox[1]: # bbox is t, r, b, l
                        with registry_lock:
                            p = person_registry.get(bid)
                            if not p: continue
                            now = time.time()
                            if recog_queue.full(): continue
                            if p["status"] == "visitor" and (now - p["last_recog"]) < UNKNOWN_LOCKOUT_TIME: continue
                            if p["status"] == "known" and (now - p["last_recog"]) < RECOGNITION_COOLDOWN: continue
                            if (now - p["last_recog"]) < 1.0: continue
                            
                        recog_queue.put({"id": bid, "frame": rgb, "loc": face_loc})

        # --- DRAWING (Make the final image ready for Flask) ---
        # We draw directly on 'frame_sm'
        with registry_lock:
            # Draw Bodies
            for track_id, (t, r, b, l) in body_boxes.items():
                p = person_registry.get(track_id, {})
                name = p.get("name", "Unknown")
                status = p.get("status", "tracking")
                
                color = (100, 100, 255) # Red (Unknown)
                if status == "visitor": color = (150, 150, 150) # Gray
                elif name != "Unknown": color = (100, 255, 100) # Green
                
                cv2.rectangle(frame_sm, (l, t), (r, b), color, 2)
                cv2.putText(frame_sm, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            # Draw Faces
            for (t, r, b, l) in curr_faces:
                cv2.rectangle(frame_sm, (l, t), (r, b), (255, 255, 0), 2)
                
            # Draw Timestamp (Proof of Life)
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame_sm, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode ONCE and store bytes for Flask to grab
        (flag, encodedImage) = cv2.imencode(".jpg", frame_sm, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if flag:
            latest_jpeg_bytes = encodedImage.tobytes()

# --- MAIN ---
def _load_resources():
    global yolo_body_model, yolo_face_model
    logger.info("Loading YOLO Models...")
    yolo_body_model = YOLO("yolo11n.pt", verbose=False)
    yolo_body_model.to(DEVICE_STR)
    face_path = get_face_model_path()
    if face_path:
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)
    load_known_faces(KNOWN_FACES_DIR)

if __name__ == "__main__":
    print("--- SYSTEM STARTING (ZERO-LATENCY MODE) ---")
    
    # 1. Start Reader
    stream_reader = FastVideoReader(STREAM_PI_RTSP).start()
    
    # 2. Load
    _load_resources()
    
    # 3. Threads
    threading.Thread(target=recognition_worker_thread, daemon=True).start()
    threading.Thread(target=video_processing_thread, daemon=True).start()
    
    # 4. Flask
    threading.Thread(target=run_flask, daemon=True).start()
    
    print("--- READY ---")
    ips = get_ip_addresses()
    for ip in ips: print(f" -> http://{ip}:{WEB_SERVER_PORT}/")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        APP_SHOULD_QUIT = True
        stream_reader.stop()
        print("Stopping...")