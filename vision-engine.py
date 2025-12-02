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
import queue
from concurrent.futures import ThreadPoolExecutor
import torch
from flask import Flask, Response
from ultralytics import YOLO

# --- CONFIGURATION ---
RECOGNITION_TOLERANCE = 0.5 
WEB_SERVER_PORT = 5005
BOX_PADDING = 40
DIAGNOSTICS_INTERVAL = 5.0 
HIGH_CONFIDENCE_THRESHOLD = 65.0 
RECOGNITION_COOLDOWN = 2.0 
UNKNOWN_RETRY_LIMIT = 5          
UNKNOWN_LOCKOUT_TIME = 30.0      

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

# --- FLASK APP ---
app = Flask(__name__)
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

# --- SETTINGS ---
STREAM_PI_IP = "100.114.210.58"
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp"
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 
FACE_RECOGNITION_NTH_FRAME = 3 

COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255)
COLOR_BODY_VISITOR = (150, 150, 150) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_FG = (255, 255, 255)

YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt"}

# --- GLOBAL STATE ---
video_lock = threading.Lock()
results_lock = threading.Lock()
registry_lock = threading.Lock()

latest_raw_frame = None       
latest_processed_results = {  
    "body_boxes": {},
    "active_ids": [],
    "face_boxes": [],
}

# The Registry: The Source of Truth for Names
person_registry = {} 

# The Queue: Jobs for the slow CPU thread
recog_queue = queue.Queue(maxsize=1) 

APP_SHOULD_QUIT = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

perf_stats = {
    "fps": 0, "body_ms": 0, "face_det_ms": 0, "face_rec_ms": 0,
    "last_print": time.time(), "frame_counter": 0, "active_jobs": 0
}

server_data = {
    "yolo_model_key": "n", 
    "yolo_conf": 0.4
}

# --- RESOURCES ---
known_face_encodings = []
known_face_names = []
yolo_body_model = None
yolo_face_model = None

# --- HELPER FUNCTIONS ---
def print_diagnostics():
    global perf_stats
    now = time.time()
    if now - perf_stats["last_print"] > DIAGNOSTICS_INTERVAL:
        fps = perf_stats["frame_counter"] / (now - perf_stats["last_print"])
        vram = 0
        if torch.cuda.is_available(): vram = torch.cuda.memory_reserved(0) / 1024 / 1024
        
        print(f"\n=== DIAGNOSTICS (FPS: {fps:.2f} | VRAM: {vram:.0f}MB) ===")
        print(f" > YOLO Body:   {perf_stats['body_ms']:.1f} ms")
        print(f" > YOLO Face:   {perf_stats['face_det_ms']:.1f} ms")
        print(f" > Face Recog:  {perf_stats['face_rec_ms']:.1f} ms (Async Thread)")
        print(f" > Queue Size:  {recog_queue.qsize()}")
        print("==============================================\n")
        
        perf_stats["last_print"] = now; perf_stats["frame_counter"] = 0; perf_stats["face_rec_ms"] = 0

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
    logger.info(f"Loading faces from {known_faces_dir}...")
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

def calculate_iou(boxA, boxB):
    f_x1, f_y1, f_x2, f_y2 = boxA[3], boxA[0], boxA[1], boxA[2]
    b_x1, b_y1, b_x2, b_y2 = boxB[3], boxB[0], boxB[1], boxB[2]
    xA = max(f_x1, b_x1); yA = max(f_y1, b_y1)
    xB = min(f_x2, b_x2); yB = min(f_y2, b_y2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (f_x2 - f_x1) * (f_y2 - f_y1)
    return interArea / float(boxAArea) if boxAArea > 0 else 0

def get_best_matching_body(face_box, body_boxes):
    best_iou = 0; best_id = None
    for track_id, body_box in body_boxes.items():
        iou = calculate_iou(face_box, body_box)
        if iou > 0.5 and iou > best_iou: best_iou = iou; best_id = track_id
    return best_id

def get_ip_addresses():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80))
        return [s.getsockname()[0]]
    except: return []

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return "<html><body style='background:black; text-align:center;'><h1 style='color:white;'>Vision Engine Live</h1><img src='/video_feed' style='width:90%; border:2px solid #333;'></body></html>"

def generate_frames():
    # --- DECOUPLED RENDERING LOOP ---
    while True:
        frame = None
        with video_lock:
            if latest_raw_frame is not None: frame = latest_raw_frame.copy()
        
        if frame is None:
            time.sleep(0.02); continue

        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)

        # Get Tracking Data
        ai_data = None
        with results_lock: ai_data = latest_processed_results.copy()
        
        # Get Registry Data (Thread Safe copy)
        current_registry = {}
        with registry_lock: current_registry = person_registry.copy()

        if ai_data:
            # Draw Faces
            for (ft, fr, fb, fl) in ai_data.get("face_boxes", []):
                cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)
            
            # Draw Bodies
            for track_id in ai_data.get("active_ids", []):
                if track_id in ai_data.get("body_boxes", {}):
                    t, r, b, l = ai_data["body_boxes"][track_id]
                    
                    # Defaults
                    name = "Scanning..."; conf = 0; color = COLOR_BODY_UNKNOWN
                    
                    if track_id in current_registry:
                        p_data = current_registry[track_id]
                        name = p_data["name"]
                        conf = p_data["conf"]
                        
                        if name != "Unknown" and name != "Scanning...":
                            color = COLOR_BODY_KNOWN
                        elif p_data.get("status") == "visitor":
                            name = "Visitor"
                            color = COLOR_BODY_VISITOR
                    
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    
                    label = f"{name}"
                    if conf > 0: label += f" ({int(conf)}%)"
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
                    cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if flag:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03) 

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try:
        app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)
    except OSError:
        logger.error(f"Port {WEB_SERVER_PORT} is in use.")

# --- RESOURCES ---
def _load_resources():
    global yolo_body_model, yolo_face_model
    logger.info("Loading YOLO Models...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR)
    face_path = get_face_model_path()
    if face_path:
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)
    load_known_faces(KNOWN_FACES_DIR)

# --- THREAD 1: VIDEO READER (60+ FPS) ---
def _frame_reader_loop(source):
    global latest_raw_frame, APP_SHOULD_QUIT
    logger.info(f"Starting Video Reader on: {source}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    while not APP_SHOULD_QUIT:
        if not cap.isOpened():
            time.sleep(2); cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG); continue
        ret, frame = cap.read()
        if not ret:
            logger.warning("Stream dropped..."); cap.release(); continue
        with video_lock: latest_raw_frame = frame
        time.sleep(0.015) 
    cap.release()

# --- THREAD 2: RECOGNITION WORKER (CPU BOUND) ---
def recognition_worker_thread():
    global person_registry, perf_stats
    logger.info("Background Recognition Worker Started.")
    
    while not APP_SHOULD_QUIT:
        try:
            # Block for 0.1s so we can check APP_SHOULD_QUIT
            job = recog_queue.get(timeout=0.1) 
        except queue.Empty:
            continue

        # Job Structure: {"id": track_id, "frame": rgb_frame, "loc": face_loc}
        track_id = job["id"]
        rgb_frame = job["frame"]
        face_loc = job["loc"]
        
        t_start = time.time()
        
        # 1. ENCODING (The heavy 1.3s part)
        try:
            encs = face_recognition.face_encodings(rgb_frame, [face_loc])
        except Exception as e:
            logger.error(f"Recog Error: {e}")
            encs = []

        new_name = "Unknown"
        new_conf = 0.0

        # 2. MATCHING
        if encs and len(known_face_encodings) > 0:
            matches = face_recognition.compare_faces(known_face_encodings, encs[0], tolerance=RECOGNITION_TOLERANCE)
            dists = face_recognition.face_distance(known_face_encodings, encs[0])
            if True in matches:
                best_idx = np.argmin(dists)
                new_name = known_face_names[best_idx]
                new_conf = max(0, min(100, (1.0 - dists[best_idx]) * 100))
        
        # 3. UPDATE REGISTRY (Thread Safe)
        with registry_lock:
            if track_id in person_registry:
                entry = person_registry[track_id]
                
                if new_name != "Unknown":
                    # We found a match!
                    entry["name"] = new_name
                    entry["conf"] = new_conf
                    entry["status"] = "known"
                    entry["retries"] = 0 # Reset retries
                else:
                    # Still unknown
                    entry["retries"] += 1
                    if entry["retries"] >= UNKNOWN_RETRY_LIMIT:
                        entry["status"] = "visitor"
                        logger.info(f"ID {track_id} marked as Visitor (Stopped trying).")
                
                entry["last_recog"] = time.time()

        perf_stats["face_rec_ms"] = (time.time() - t_start) * 1000

# --- THREAD 3: TRACKING LOOP (GPU BOUND - 30 FPS) ---
def video_processing_thread():
    global latest_processed_results, perf_stats, person_registry
    frame_count = 0
    curr_faces = [] # FIXED: Initialize outside the loop to prevent UnboundLocalError

    while not APP_SHOULD_QUIT:
        frame = None
        with video_lock:
            if latest_raw_frame is not None: frame = latest_raw_frame.copy()
        
        if frame is None: time.sleep(0.01); continue
        
        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        frame_count += 1
        perf_stats["frame_counter"] += 1
        t0 = time.time()
        
        # --- YOLO BODY ---
        body_results = yolo_body_model.track(frame, persist=True, classes=[0], conf=server_data["yolo_conf"], verbose=False)
        perf_stats["body_ms"] = (time.time() - t0) * 1000
        
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
                    
                    # Initialize in Registry if new
                    if track_id not in person_registry:
                        person_registry[track_id] = {
                            "name": "Unknown", "conf": 0.0, 
                            "last_recog": 0, "retries": 0, "status": "tracking"
                        }

        # --- YOLO FACE ---
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model:
            t1 = time.time()
            face_results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
            perf_stats["face_det_ms"] = (time.time() - t1) * 1000
            
            # Reset and update face boxes
            curr_faces = []
            if len(face_results) > 0:
                for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    curr_faces.append((t, r, b, l))
            
            # --- DISPATCH TO WORKER ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for face_loc in curr_faces:
                bid = get_best_matching_body(face_loc, body_boxes)
                if bid is not None:
                    with registry_lock:
                        p = person_registry.get(bid)
                        if not p: continue
                        
                        now = time.time()
                        
                        # LOGIC: WHEN TO SEND TO WORKER?
                        if recog_queue.full(): continue
                        
                        if p["status"] == "visitor":
                            if (now - p["last_recog"]) < UNKNOWN_LOCKOUT_TIME: continue
                            else: p["status"] = "tracking"; p["retries"] = 0 
                        
                        if p["status"] == "known" and p["conf"] > HIGH_CONFIDENCE_THRESHOLD:
                             if (now - p["last_recog"]) < RECOGNITION_COOLDOWN: continue
                        
                        if (now - p["last_recog"]) < 1.0: continue

                    # Push to Queue
                    recog_queue.put({"id": bid, "frame": rgb, "loc": face_loc})

        # Publish Results for Flask
        with results_lock:
            latest_processed_results["body_boxes"] = body_boxes
            latest_processed_results["active_ids"] = active_ids
            latest_processed_results["face_boxes"] = curr_faces

        print_diagnostics()

# --- MAIN ---
if __name__ == "__main__":
    print("--- SYSTEM STARTING (ASYNC ARCHITECTURE) ---")
    
    # 1. Video Reader
    threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True).start()
    
    # 2. Load Models
    _load_resources()
    
    # 3. Recognition Worker (The Slow Thread)
    threading.Thread(target=recognition_worker_thread, daemon=True).start()

    # 4. Tracking Loop (The Fast Thread)
    threading.Thread(target=video_processing_thread, daemon=True).start()
    
    # 5. Flask Render
    threading.Thread(target=run_flask, daemon=True).start()
    
    print("--- READY ---")
    ips = get_ip_addresses()
    for ip in ips: print(f" -> http://{ip}:{WEB_SERVER_PORT}/")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        APP_SHOULD_QUIT = True
        print("Stopping...")