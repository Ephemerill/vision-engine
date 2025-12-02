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
import subprocess 
import base64
import traceback
import socket
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
import torch
from flask import Flask, Response
from ultralytics import YOLO

# --- CONFIGURATION ---
RECOGNITION_TOLERANCE = 0.5 
WEB_SERVER_PORT = 5005
BOX_PADDING = 40

# --- PERFORMANCE TUNING ---
HIGH_CONFIDENCE_THRESHOLD = 65.0 
RECOGNITION_COOLDOWN = 2.0 
DIAGNOSTICS_INTERVAL = 5.0 # Print stats every 5 seconds

# --- YOLO FACE CONFIG ---
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
    logger.warning("⚠️  CRITICAL: GPU NOT DETECTED. Running on CPU.")

# --- IMPORTS (PYTORCH ONLY) ---
# DISABLED GFPGAN
GFPGAN_AVAILABLE = False

# --- FLASK APP ---
app = Flask(__name__)
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

# --- SETTINGS ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_OFF = 'off'

STREAM_PI_IP = "100.114.210.58"
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp"
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"
STREAM_WEBCAM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 
FACE_RECOGNITION_NTH_FRAME = 3 

COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_FG = (255, 255, 255)

# Default Body Models
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt"}

# --- GLOBAL STATE ---
data_lock = threading.Lock()
output_frame = None
latest_raw_frame = None 
APP_SHOULD_QUIT = False
VIDEO_THREAD_STARTED = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

# --- METRICS STATE ---
perf_stats = {
    "fps": 0,
    "body_ms": 0,
    "face_det_ms": 0,
    "face_rec_ms": 0,
    "last_print": time.time(),
    "frame_counter": 0,
    "start_time": time.time()
}

server_data = {
    "is_recording": False, "keyframe_count": 0, "action_result": "", "live_faces": [],
    "model": MODEL_GEMMA, 
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
def print_diagnostics():
    global perf_stats
    now = time.time()
    if now - perf_stats["last_print"] > DIAGNOSTICS_INTERVAL:
        fps = perf_stats["frame_counter"] / (now - perf_stats["last_print"])
        
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.memory_reserved(0) / 1024 / 1024 # MB
        
        print("\n=== PERFORMANCE DIAGNOSTICS ===")
        print(f" > FPS:         {fps:.2f}")
        print(f" > VRAM Used:   {vram_usage:.0f} MB")
        print(f" > YOLO Body:   {perf_stats['body_ms']:.1f} ms")
        print(f" > YOLO Face:   {perf_stats['face_det_ms']:.1f} ms")
        print(f" > Face Recog:  {perf_stats['face_rec_ms']:.1f} ms (CPU)")
        print("===============================\n")
        
        # Reset counters
        perf_stats["last_print"] = now
        perf_stats["frame_counter"] = 0
        perf_stats["face_rec_ms"] = 0 # Reset specifically as it doesn't run every frame

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

def calculate_iou(boxA, boxB):
    f_x1, f_y1, f_x2, f_y2 = boxA[3], boxA[0], boxA[1], boxA[2]
    b_x1, b_y1, b_x2, b_y2 = boxB[3], boxB[0], boxB[1], boxB[2]

    xA = max(f_x1, b_x1)
    yA = max(f_y1, b_y1)
    xB = min(f_x2, b_x2)
    yB = min(f_y2, b_y2)

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = (f_x2 - f_x1) * (f_y2 - f_y1)
    if boxAArea == 0: return 0
    return interArea / float(boxAArea)

def get_best_matching_body(face_box, body_boxes):
    best_iou = 0
    best_id = None
    for track_id, body_box in body_boxes.items():
        iou = calculate_iou(face_box, body_box)
        if iou > 0.5 and iou > best_iou:
            best_iou = iou
            best_id = track_id
    return best_id

def get_ip_addresses():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80)) 
        ips.append(s.getsockname()[0])
        s.close()
    except: pass
    return list(set(ips))

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return "<html><body style='background:black; text-align:center;'><h1 style='color:white;'>Vision Engine Live</h1><img src='/video_feed' style='width:90%; border:2px solid #333;'></body></html>"

def generate_frames():
    while True:
        with data_lock:
            if output_frame is None:
                time.sleep(0.1); continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try:
        app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)
    except OSError:
        logger.error(f"Port {WEB_SERVER_PORT} is in use.")

# --- RESOURCE LOADING ---
def _load_resources():
    global GFPGANer, GFPGAN_AVAILABLE, yolo_body_model, yolo_face_model, gfpgan_enhancer
    
    logger.info("Loading YOLO11 Body Model (GPU)...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR) 

    face_path = get_face_model_path()
    if face_path:
        logger.info(f"Loading Face Model: {face_path} (GPU)...")
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)
    else:
        logger.error("Failed to load Face Model.")

    # GFPGAN is DISABLED in this version

    load_known_faces(KNOWN_FACES_DIR)

# --- VIDEO THREADS ---
def _frame_reader_loop(source):
    global latest_raw_frame, data_lock, APP_SHOULD_QUIT, VIDEO_THREAD_STARTED, CURRENT_STREAM_SOURCE
    cap = None
    
    def connect(src):
        if isinstance(src, int): return cv2.VideoCapture(src)
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG) 

    logger.info(f"Starting Video Reader on: {source}")
    while not APP_SHOULD_QUIT:
        if cap is None or not cap.isOpened():
            cap = connect(source)
            if not cap.isOpened() and source == STREAM_PI_RTSP:
                logger.warning("RTSP Failed. Switching to HLS Fallback.")
                source = STREAM_PI_HLS; CURRENT_STREAM_SOURCE = STREAM_PI_HLS; continue
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); VIDEO_THREAD_STARTED = True
                logger.info("Video Stream Connected.")
            else: 
                time.sleep(2); continue

        ret, frame = cap.read()
        if not ret: 
            logger.warning("Frame dropped. Reconnecting...")
            cap.release(); time.sleep(0.5); continue
        with data_lock: latest_raw_frame = frame
    if cap: cap.release()

def video_processing_thread():
    global data_lock, output_frame, server_data, APP_SHOULD_QUIT, latest_raw_frame
    global person_registry, last_face_locations, perf_stats
    
    frame_count = 0

    while not APP_SHOULD_QUIT:
        frame = None
        with data_lock:
            if latest_raw_frame is not None: frame = latest_raw_frame.copy()
        
        if frame is None: 
            time.sleep(0.01)
            continue

        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        if frame is None: continue
        frame_count += 1
        perf_stats["frame_counter"] += 1
        
        with data_lock:
            enhancement_mode = server_data["face_enhancement_mode"]
            yolo_conf = server_data["yolo_conf"]

        # 1. YOLO Body Tracking
        t_start_body = time.time()
        body_results = yolo_body_model.track(frame, persist=True, classes=[0], conf=yolo_conf, verbose=False)
        perf_stats["body_ms"] = (time.time() - t_start_body) * 1000

        body_boxes = {}; active_track_ids = []
        
        if body_results[0].boxes.id is not None:
            boxes = body_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = body_results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                body_boxes[track_id] = (t, r, b, l) 
                active_track_ids.append(track_id)
                if track_id not in person_registry:
                    person_registry[track_id] = {
                        "name": "Unknown", 
                        "conf": 0.0, 
                        "last_seen": time.time(),
                        "last_recog_time": 0.0
                    }

        # 2. YOLO Face Detection & Recognition
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model:
            t_start_face = time.time()
            face_results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
            perf_stats["face_det_ms"] = (time.time() - t_start_face) * 1000
            
            current_face_locations = []
            if len(face_results) > 0:
                for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    current_face_locations.append((t, r, b, l)) 
            
            last_face_locations = current_face_locations
            rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for face_loc in current_face_locations:
                body_id = get_best_matching_body(face_loc, body_boxes)
                
                if body_id is not None:
                    current_data = person_registry[body_id]
                    now = time.time()
                    
                    if (current_data["name"] != "Unknown" and 
                        current_data["conf"] > HIGH_CONFIDENCE_THRESHOLD and 
                        (now - current_data["last_recog_time"]) < RECOGNITION_COOLDOWN):
                        person_registry[body_id]["last_seen"] = now
                        continue 

                    top, right, bottom, left = face_loc
                    input_image_encoding = rgb_frame_for_encoding
                    encoding_loc = [face_loc]

                    # --- TIMING RECOGNITION ---
                    t_start_rec = time.time()
                    face_enc = face_recognition.face_encodings(input_image_encoding, encoding_loc)
                    
                    new_name = "Unknown"; new_conf = 0.0
                    
                    if face_enc and len(known_face_encodings) > 0:
                        matches = face_recognition.compare_faces(known_face_encodings, face_enc[0], tolerance=RECOGNITION_TOLERANCE)
                        dists = face_recognition.face_distance(known_face_encodings, face_enc[0])
                        if True in matches:
                            best_idx = np.argmin(dists)
                            new_name = known_face_names[best_idx]
                            new_conf = max(0, min(100, (1.0 - dists[best_idx]) * 100))
                    
                    # Update Recog Stats (accumulate if multiple faces, though usually 1)
                    perf_stats["face_rec_ms"] = (time.time() - t_start_rec) * 1000

                    current_name = current_data["name"]
                    current_conf = current_data["conf"]
                    should_update = False
                    
                    if new_name != "Unknown":
                        if current_name == "Unknown": should_update = True
                        elif new_name == current_name:
                            if new_conf > current_conf - 10: should_update = True
                        else:
                            if new_conf > current_conf + 15: should_update = True
                    else:
                        if current_name != "Unknown" and current_conf < 40: should_update = True

                    if should_update:
                        person_registry[body_id]["name"] = new_name
                        person_registry[body_id]["conf"] = new_conf
                    
                    person_registry[body_id]["last_seen"] = now
                    person_registry[body_id]["last_recog_time"] = now

        # 4. Drawing
        for (ft, fr, fb, fl) in last_face_locations:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

        live_face_payload = []
        for track_id in active_track_ids:
            t, r, b, l = body_boxes[track_id]
            data = person_registry.get(track_id, {"name": "Unknown", "conf": 0})
            name = data["name"]; conf = data["conf"]
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            label = f"{name}"
            if name != "Unknown": label += f" ({int(conf)}%)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
            cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)
            live_face_payload.append({"name": name, "confidence": conf})

        # --- DIAGNOSTICS PRINT ---
        print_diagnostics()

        with data_lock:
            output_frame = frame
            server_data["live_faces"] = live_face_payload

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" HEADLESS VISION ENGINE STARTING (WITH DIAGNOSTICS) ")
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
            time.sleep(1) # Main loop just sleeps, work is in threads
    except KeyboardInterrupt:
        APP_SHOULD_QUIT = True
        print("\nShutting down...")