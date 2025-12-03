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
import logging
import requests
import socket
import base64
import torch

# --- NEW IMPORTS FOR WEBSOCKETS ---
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from ultralytics import YOLO

# --- CONFIGURATION ---
RECOGNITION_TOLERANCE = 0.5 
WEB_SERVER_PORT = 5005

# --- MODEL SELECTION (YOLOv11 Face) ---
FACE_MODEL_NAME = "yolov11n-face.pt"
FACE_MODEL_URL = f"https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/{FACE_MODEL_NAME}"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisionEngine")

# --- HARDWARE CHECK ---
if torch.cuda.is_available():
    DEVICE_STR = 'cuda:0'
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"✅ GPU DETECTED: {gpu_name}")
elif torch.backends.mps.is_available():
    DEVICE_STR = 'mps'
    logger.info(f"✅ MAC GPU (MPS) DETECTED")
else:
    DEVICE_STR = 'cpu'
    logger.warning("⚠️  CRITICAL: GPU NOT DETECTED. Running on CPU.")

# --- IMPORTS (PYTORCH ONLY) ---
try:
    import gfpgan
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

# --- FLASK & SOCKETIO APP ---
app = Flask(__name__)
# async_mode='threading' is crucial for compatibility with OpenCV on macOS/Windows
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

# --- SETTINGS ---
STREAM_PI_IP = "100.114.210.58"
# SWITCH TO UDP: 'udp' drops lost packets instead of pausing to retry (low latency)
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam" 
STREAM_WEBCAM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 
FACE_RECOGNITION_NTH_FRAME = 3 
BOX_PADDING = 10

COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_FG = (255, 255, 255)

YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt"}

# --- GLOBAL STATE ---
data_lock = threading.Lock()
# OPTIMIZATION: Atomic Packet [Frame, Timestamp]
latest_packet = [None, 0.0] 

APP_SHOULD_QUIT = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

server_data = {
    "live_faces": [],
    "yolo_model_key": "n", 
    "yolo_conf": 0.4, 
    "face_enhancement_mode": "off" 
}

if not GFPGAN_AVAILABLE:
    server_data["face_enhancement_mode"] = "off_disabled"

# --- RESOURCES ---
known_face_encodings = []
known_face_names = []
person_registry = {} 
last_face_locations = [] 

GFPGANer = None
yolo_body_model = None
yolo_face_model = None
gfpgan_enhancer = None

# --- HELPER FUNCTIONS ---
def get_face_model_path():
    if os.path.exists(FACE_MODEL_NAME): return FACE_MODEL_NAME
    logger.info(f"Downloading New Face Model ({FACE_MODEL_NAME})...")
    try:
        response = requests.get(FACE_MODEL_URL, stream=True)
        with open(FACE_MODEL_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
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
    logger.info(f"Loading faces from {known_faces_dir}...")
    count = 0
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
                        count += 1
                except: pass
    logger.info(f"Loaded {count} known face identities.")

def get_containing_body_box(face_box, body_boxes):
    ft, fr, fb, fl = face_box
    cx, cy = (fl + fr) / 2, (ft + fb) / 2
    for track_id, (bt, br, bb, bl) in body_boxes.items():
        if bl < cx < br and bt < cy < bb: return track_id
    return None

def get_ip_addresses():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80)) 
        ips.append(s.getsockname()[0])
        s.close()
    except: pass
    return list(set(ips))

def draw_perf_overlay(frame, timings, lag_ms):
    cv2.rectangle(frame, (0, 0), (160, 140), (0, 0, 0), -1)
    lag_color = (0, 255, 0) 
    if lag_ms > 200: lag_color = (0, 165, 255) 
    if lag_ms > 1000: lag_color = (0, 0, 255)  
    cv2.putText(frame, f"LAG: {lag_ms:.0f} ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lag_color, 2)
    
    y = 40
    for k, v in timings.items():
        cv2.putText(frame, f"{k}: {v:.1f}ms", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        y += 15
    return frame

# --- FLASK ROUTES ---
@app.route('/')
def index():
    # WEBSOCKET CLIENT HTML
    # This Javascript listens for 'frame_data' events and updates the Image SRC instantly.
    # No buffering.
    return render_template_string("""
    <html>
    <head>
        <title>Vision Engine (WebSocket)</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { background: black; color: white; text-align: center; font-family: sans-serif; }
            #video-container { margin-top: 20px; }
            img { border: 2px solid #333; max-width: 90%; }
            #status { color: #888; font-size: 12px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>Vision Engine (Low Latency)</h1>
        <div id="video-container">
            <img id="live-feed" src="" alt="Waiting for stream..." />
        </div>
        <div id="status">Connecting...</div>

        <script>
            var socket = io();
            var img = document.getElementById('live-feed');
            var statusDiv = document.getElementById('status');
            var startTime = 0;

            socket.on('connect', function() {
                statusDiv.innerText = "Connected via WebSocket";
            });

            socket.on('frame_data', function(msg) {
                // Instantly swap the image source
                img.src = "data:image/jpeg;base64," + msg.image;
            });
            
            socket.on('disconnect', function() {
                statusDiv.innerText = "Disconnected";
            });
        </script>
    </body>
    </html>
    """)

# --- RESOURCE LOADING ---
def _load_resources():
    global GFPGANer, GFPGAN_AVAILABLE, yolo_body_model, yolo_face_model, gfpgan_enhancer
    
    logger.info("Loading YOLO Body...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR) 

    face_path = get_face_model_path()
    if face_path:
        logger.info("Loading YOLO Face...")
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)

    if GFPGAN_AVAILABLE:
        logger.info("Loading GFPGAN...")
        try:
            gfpgan_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE_STR)
        except: 
            server_data["face_enhancement_mode"] = "off_disabled"
    load_known_faces(KNOWN_FACES_DIR)

# --- SPEED READER THREAD (INPUT SIDE FIX) ---
def _frame_reader_loop(source):
    global latest_packet, APP_SHOULD_QUIT
    
    # 1. SWITCH TO UDP to stop retransmission lag
    # 2. NOBUFFER flag for FFmpeg
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
    
    logger.info(f"Starting Reader on: {source} (UDP Mode)")
    
    cap = None
    while not APP_SHOULD_QUIT:
        if cap is None or not cap.isOpened():
            if isinstance(source, int): cap = cv2.VideoCapture(source)
            else: cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 0) # Attempt zero buffer
                logger.info("✅ Stream Connected")
            else:
                time.sleep(2); continue

        ret, frame = cap.read()
        if not ret: 
            logger.warning("Frame dropped/Stream lost.")
            cap.release(); cap = None; time.sleep(0.5); continue
        
        # ATOMIC OVERWRITE 
        with data_lock: 
            latest_packet[0] = frame
            latest_packet[1] = time.time()
            
    if cap: cap.release()

# --- PROCESSING THREAD (OUTPUT SIDE FIX) ---
def video_processing_thread():
    global latest_packet, person_registry, last_face_locations
    
    frame_count = 0
    
    while not APP_SHOULD_QUIT:
        t_start_proc = time.perf_counter()
        timings = {}
        
        # 1. ACQUIRE
        frame = None
        capture_ts = 0
        with data_lock:
            if latest_packet[0] is not None: 
                frame = latest_packet[0].copy()
                capture_ts = latest_packet[1]
        
        if frame is None: 
            time.sleep(0.005)
            continue

        # LAG CALCULATION
        lag_ms = (time.time() - capture_ts) * 1000

        # 2. RESIZE
        t0 = time.perf_counter()
        frame = resize_with_aspect_ratio(frame)
        timings['resize'] = (time.perf_counter() - t0) * 1000
        frame_count += 1
        
        yolo_conf = server_data["yolo_conf"]

        # 3. YOLO BODY
        t0 = time.perf_counter()
        body_results = yolo_body_model.track(frame, persist=True, classes=[0], conf=yolo_conf, verbose=False)
        body_boxes = {}; active_track_ids = []
        
        if body_results[0].boxes.id is not None:
            boxes = body_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = body_results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                body_boxes[track_id] = (t, r, b, l)
                active_track_ids.append(track_id)
                if track_id not in person_registry:
                    person_registry[track_id] = {"name": "Unknown", "conf": 0.0, "last_seen": time.time()}
        timings['body'] = (time.perf_counter() - t0) * 1000

        # 4. YOLO FACE
        t0 = time.perf_counter()
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model:
            face_results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
            current_face_locations = []
            if len(face_results) > 0:
                for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    current_face_locations.append((t, r, b, l))
            last_face_locations = current_face_locations
        timings['face'] = (time.perf_counter() - t0) * 1000

        # 5. RECOGNITION
        t0 = time.perf_counter()
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for face_loc in last_face_locations:
                body_id = get_containing_body_box(face_loc, body_boxes)
                if body_id is not None:
                    # Simple crop for speed, full enhancement disabled for latency test
                    face_enc = face_recognition.face_encodings(rgb, [face_loc])
                    name = "Unknown"; conf = 0.0
                    if face_enc and len(known_face_encodings) > 0:
                        matches = face_recognition.compare_faces(known_face_encodings, face_enc[0], tolerance=RECOGNITION_TOLERANCE)
                        dists = face_recognition.face_distance(known_face_encodings, face_enc[0])
                        if True in matches:
                            best_idx = np.argmin(dists)
                            name = known_face_names[best_idx]
                            conf = max(0, min(100, (1.0 - dists[best_idx]) * 100))
                    
                    if name != "Unknown":
                        person_registry[body_id]["name"] = name
                        person_registry[body_id]["conf"] = conf
        timings['recog'] = (time.perf_counter() - t0) * 1000

        # 6. DRAW
        t0 = time.perf_counter()
        for (ft, fr, fb, fl) in last_face_locations:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

        live_face_payload = []
        for track_id in active_track_ids:
            t, r, b, l = body_boxes[track_id]
            data = person_registry.get(track_id, {"name": "Unknown", "conf": 0})
            name = data["name"]
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"{track_id}: {name}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            live_face_payload.append({"name": name})
        timings['draw'] = (time.perf_counter() - t0) * 1000
        timings['total'] = (time.perf_counter() - t_start_proc) * 1000
        
        frame = draw_perf_overlay(frame, timings, lag_ms)
        
        # 7. WEBSOCKET EMIT (THE OUTPUT FIX)
        # Encode to JPEG, then Base64, then push immediately to client.
        # This bypasses HTTP stream buffers.
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        b64_image = base64.b64encode(buffer).decode('utf-8')
        
        socketio.emit('frame_data', {'image': b64_image})
        
        with data_lock:
            server_data["live_faces"] = live_face_payload
            
        # Optional: Sleep slightly to cap at ~30 FPS to save CPU
        time.sleep(0.01) 

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" VISION ENGINE (LOW LATENCY WEBSOCKET MODE) ")
    print("---------------------------------------------------")
    
    reader = threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True)
    reader.start()
    
    _load_resources()
    
    proc = threading.Thread(target=video_processing_thread, daemon=True)
    proc.start()
    
    ips = get_ip_addresses()
    for ip in ips:
        print(f" -> http://{ip}:{WEB_SERVER_PORT}/")

    try:
        # Use socketio.run instead of app.run
        socketio.run(app, host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        APP_SHOULD_QUIT = True