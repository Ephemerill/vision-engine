import warnings
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
from concurrent.futures import ThreadPoolExecutor
import torch
from flask import Flask, Response
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
else:
    DEVICE_STR = 'cpu'
    logger.warning("⚠️ CRITICAL: GPU NOT DETECTED. Running on CPU.")

# --- IMPORTS (PYTORCH ONLY) ---
try:
    import gfpgan
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

# --- FLASK APP ---
app = Flask(__name__)
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

# --- SETTINGS ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_OFF = 'off'

STREAM_PI_IP = "100.114.210.58"
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam"
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"
STREAM_WEBCAM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5
FACE_RECOGNITION_NTH_FRAME = 5  # Increased from 3 for better performance
BOX_PADDING = 10

COLOR_BODY_KNOWN = (255, 100, 100)
COLOR_BODY_UNKNOWN = (100, 100, 255)
COLOR_FACE_BOX = (255, 255, 0)
COLOR_TEXT_FG = (255, 255, 255)

# Default Body Models (Standard YOLO11)
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt", "x": "yolo11x.pt"}

# --- GLOBAL STATE ---
data_lock = threading.Lock()
output_frame = None
output_frame_encoded = None  # NEW: Pre-encoded JPEG buffer

# OPTIMIZATION: Latest frame packet
latest_packet = [None, 0.0]

APP_SHOULD_QUIT = False
VIDEO_THREAD_STARTED = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP

server_data = {
    "is_recording": False, "keyframe_count": 0, "action_result": "", "live_faces": [],
    "model": MODEL_GEMMA,
    "yolo_model_key": "n",
    "yolo_conf": 0.35,  # Lowered from 0.4 for faster processing
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
    if os.path.exists(FACE_MODEL_NAME):
        return FACE_MODEL_NAME
    
    logger.info(f"Downloading New Face Model ({FACE_MODEL_NAME})...")
    try:
        response = requests.get(FACE_MODEL_URL, stream=True)
        if response.status_code != 200:
            logger.error(f"Download failed: Status {response.status_code}")
            return None
        
        with open(FACE_MODEL_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")
        return FACE_MODEL_NAME
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None

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
                image_path = os.path.join(person_dir, filename)
                try:
                    img = face_recognition.load_image_file(image_path)
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
    import socket
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except: pass
    return list(set(ips))

def draw_perf_overlay(frame, timings, lag_ms):
    """Draws small performance metrics in top left."""
    x, y = 10, 15
    line_h = 12
    font_scale = 0.4
    color = (0, 255, 0)
    
    cv2.rectangle(frame, (0, 0), (140, 120), (0, 0, 0), -1)
    
    lag_color = (0, 255, 0) if lag_ms < 200 else (0, 165, 255)
    if lag_ms > 1000: lag_color = (0, 0, 255)
    cv2.putText(frame, f"LAG: {lag_ms:.0f} ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lag_color, 1)
    y += 18
    
    cv2.putText(frame, f"Total Proc: {timings['total']:.1f}ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    y += line_h
    cv2.putText(frame, "----------------", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100,100,100), 1)
    y += line_h
    
    keys = ["resize", "body", "face", "recog", "enhance", "draw", "encode"]
    for k in keys:
        if k in timings:
            val = timings[k]
            cv2.putText(frame, f"{k.title()}: {val:.1f}ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200,200,200), 1)
            y += line_h
    
    return frame

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return "<html><body style='background:black; text-align:center;'><h1 style='color:white;'>Vision Engine Live</h1><img src='/video_feed' style='width:90%; border:2px solid #333;'></body></html>"

def generate_frames():
    """OPTIMIZED: Use pre-encoded frames to avoid encoding on every client request"""
    last_frame_id = None
    while True:
        with data_lock:
            if output_frame_encoded is None:
                time.sleep(0.01)
                continue
            
            # Only send if frame changed
            current_frame_id = id(output_frame_encoded)
            if current_frame_id == last_frame_id:
                time.sleep(0.005)
                continue
                
            frame_bytes = output_frame_encoded
            last_frame_id = current_frame_id
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try:
        app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False, threaded=True)
    except OSError:
        logger.error(f"Port {WEB_SERVER_PORT} is in use. Please change WEB_SERVER_PORT or kill the process.")

# --- RESOURCE LOADING ---
def _load_resources():
    global GFPGANer, GFPGAN_AVAILABLE, yolo_body_model, yolo_face_model, gfpgan_enhancer
    
    logger.info("Loading GFPGAN...")
    try:
        from gfpgan import GFPGANer as G; GFPGANer = G
        GFPGAN_AVAILABLE = True
    except: GFPGAN_AVAILABLE = False
    
    # 1. Body Tracking (Standard YOLO11)
    logger.info("Loading YOLO11 Body Model (GPU)...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR)
    
    # 2. Face Detection (YOLOv11-Face)
    face_path = get_face_model_path()
    if face_path:
        logger.info(f"Loading Face Model: {face_path} (GPU)...")
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)
    else:
        logger.error("Failed to load Face Model.")
    
    # 3. Face Enhancement (disabled by default for performance)
    if GFPGAN_AVAILABLE:
        logger.info("GFPGAN available but disabled for performance")
        server_data["face_enhancement_mode"] = "off"
    
    load_known_faces(KNOWN_FACES_DIR)

# --- SPEED READER THREAD ---
def _frame_reader_loop(source):
    global latest_packet, data_lock, APP_SHOULD_QUIT, VIDEO_THREAD_STARTED, CURRENT_STREAM_SOURCE
    
    # CRITICAL: Ultra-low latency settings
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer+discardcorrupt|flags;low_delay|max_delay;0|reorder_queue_size;0"
    
    logger.info(f"Starting Speed Reader on: {source}")
    
    cap = None
    consecutive_failures = 0
    
    while not APP_SHOULD_QUIT:
        if cap is None or not cap.isOpened():
            if isinstance(source, int): 
                cap = cv2.VideoCapture(source)
            else: 
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                # Additional low-latency settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            
            if cap.isOpened():
                VIDEO_THREAD_STARTED = True
                logger.info("✅ Stream Connected (Ultra Low Latency Mode)")
                consecutive_failures = 0
            else:
                time.sleep(2)
                continue
        
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                logger.warning("Multiple frame drops. Reconnecting...")
                cap.release()
                cap = None
                consecutive_failures = 0
            time.sleep(0.01)
            continue
        
        consecutive_failures = 0
        
        # ATOMIC OVERWRITE - Only keep newest frame
        with data_lock:
            latest_packet[0] = frame
            latest_packet[1] = time.time()
    
    if cap: cap.release()

# --- PROCESSING THREAD (Optimized) ---
def video_processing_thread():
    global data_lock, output_frame, output_frame_encoded, server_data, APP_SHOULD_QUIT, latest_packet
    global person_registry, last_face_locations
    
    frame_count = 0
    last_process_time = time.time()
    
    # JPEG encoding parameters for speed
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # Lower quality = faster encoding
    
    while not APP_SHOULD_QUIT:
        t_start_proc = time.perf_counter()
        timings = {}
        
        # 1. ACQUIRE (Non-blocking)
        frame = None
        capture_ts = 0
        with data_lock:
            if latest_packet[0] is not None:
                frame = latest_packet[0].copy()
                capture_ts = latest_packet[1]
                latest_packet[0] = None  # Clear after reading to avoid reprocessing
        
        if frame is None: 
            time.sleep(0.001)
            continue
        
        # CALC REAL-TIME LAG
        lag_ms = (time.time() - capture_ts) * 1000
        
        # CRITICAL: Drop frames if we're falling behind
        current_time = time.time()
        if lag_ms > 500:  # If more than 500ms behind, skip processing
            logger.warning(f"Dropping frame - lag: {lag_ms:.0f}ms")
            continue
        
        # 2. RESIZE
        t0 = time.perf_counter()
        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        timings['resize'] = (time.perf_counter() - t0) * 1000
        if frame is None: continue
        frame_count += 1
        
        with data_lock:
            enhancement_mode = server_data["face_enhancement_mode"]
            yolo_conf = server_data["yolo_conf"]
        
        # 3. YOLO BODY (every frame for smooth tracking)
        t0 = time.perf_counter()
        body_results = yolo_body_model.track(frame, persist=True, classes=[0], conf=yolo_conf, verbose=False)
        body_boxes = {}
        active_track_ids = []
        
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
        
        # 4. YOLO FACE (only on Nth frame)
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
        
        # 5. RECOGNITION (only on Nth frame, no enhancement for speed)
        t0 = time.perf_counter()
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and last_face_locations:
            rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for face_loc in last_face_locations:
                body_id = get_containing_body_box(face_loc, body_boxes)
                if body_id is not None:
                    top, right, bottom, left = face_loc
                    
                    # Skip GFPGAN for performance
                    encoding_loc = [face_loc]
                    face_enc = face_recognition.face_encodings(rgb_frame_for_encoding, encoding_loc)
                    
                    name = "Unknown"
                    conf = 0.0
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
                        person_registry[body_id]["last_seen"] = time.time()
        timings['recog'] = (time.perf_counter() - t0) * 1000
        
        # 6. DRAWING
        t0 = time.perf_counter()
        for (ft, fr, fb, fl) in last_face_locations:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)
        
        live_face_payload = []
        for track_id in active_track_ids:
            t, r, b, l = body_boxes[track_id]
            data = person_registry.get(track_id, {"name": "Unknown", "conf": 0})
            name = data["name"]
            conf = data["conf"]
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = f"{track_id}: {name}"
            if name != "Unknown": label += f" ({int(conf)}%)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
            cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)
            live_face_payload.append({"name": name, "confidence": conf})
        timings['draw'] = (time.perf_counter() - t0) * 1000
        
        # Calculate total timing (before overlay so it's accurate)
        timings['total'] = (time.perf_counter() - t_start_proc) * 1000
        
        # RENDER DEBUG OVERLAY (must be before encoding)
        frame = draw_perf_overlay(frame, timings, lag_ms)
        
        # 7. ENCODE ONCE (after overlay is drawn)
        t0 = time.perf_counter()
        (flag, encodedImage) = cv2.imencode(".jpg", frame, encode_param)
        timings['encode'] = (time.perf_counter() - t0) * 1000
        
        # Update global state
        if flag:
            with data_lock:
                output_frame = frame
                output_frame_encoded = bytearray(encodedImage)
                server_data["live_faces"] = live_face_payload
        
        last_process_time = current_time

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" VISION ENGINE STARTING (OPTIMIZED FOR LOW LATENCY)")
    print("---------------------------------------------------")
    
    reader = threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True)
    reader.start()
    
    _load_resources()
    
    proc = threading.Thread(target=video_processing_thread, daemon=True)
    proc.start()
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("---------------------------------------------------")
    print(" SYSTEM READY ")
    print(f" View Live Feed at:")
    
    ips = get_ip_addresses()
    for ip in ips:
        print(f" -> http://{ip}:{WEB_SERVER_PORT}/")
    print("---------------------------------------------------")
    
    try:
        while True:
            time.sleep(5)
            with data_lock: faces = server_data['live_faces']
            if faces:
                names = [f['name'] for f in faces]
                logger.info(f"Tracking: {names}")
            else:
                print(".", end="", flush=True)
    except KeyboardInterrupt:
        APP_SHOULD_QUIT = True
        print("\nShutting down...")