import os
import warnings

# --- GPU MEMORY SETTINGS (Must be first) ---
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Silence specific library warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import face_recognition 
import numpy as np
import ollama
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

# --- IMPORTS ---
try:
    import gfpgan
    GFPGAN_AVAILABLE = True
except ImportError:
    GFPGAN_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    logger.error("DeepFace not found. pip install deepface tf-keras")
    DEEPFACE_AVAILABLE = False

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

# Standard YOLO models (for Body)
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt"}

# --- GLOBAL STATE ---
data_lock = threading.Lock()
output_frame = None
latest_raw_frame = None 
APP_SHOULD_QUIT = False
VIDEO_THREAD_STARTED = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

server_data = {
    "is_recording": False, "keyframe_count": 0, "action_result": "", "live_faces": [],
    "model": MODEL_GEMMA, 
    "yolo_model_key": "l", # Defaulting to Large for better accuracy
    "yolo_conf": 0.4, 
    "face_enhancement_mode": "on" 
}

if not GFPGAN_AVAILABLE:
    server_data["face_enhancement_mode"] = "off_disabled"

# --- RESOURCES ---
known_face_db = {} 
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
    logger.info(f"Downloading Face Model ({FACE_MODEL_NAME})...")
    try:
        response = requests.get(FACE_MODEL_URL, stream=True)
        with open(FACE_MODEL_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return FACE_MODEL_NAME
    except Exception: return None

def resize_with_aspect_ratio(frame, max_w=640, max_h=480):
    if frame is None: return None
    h, w = frame.shape[:2]
    if w == 0 or h == 0: return frame
    if w <= max_w and h <= max_h: return frame
    r = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

def load_known_faces(known_faces_dir):
    global known_face_db
    if not os.path.exists(known_faces_dir): return
    
    known_face_db = {}
    logger.info(f"Loading/Embedding faces with ArcFace...")
    count = 0
    
    if not DEEPFACE_AVAILABLE: return

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
        
        if person_name not in known_face_db:
            known_face_db[person_name] = []

        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(person_dir, filename)
                try:
                    # UPDATED: Use 'opencv' backend to find the face in the reference photo.
                    # 'skip' was causing issues if the reference image wasn't already cropped.
                    embedding_objs = DeepFace.represent(
                        img_path=image_path,
                        model_name="ArcFace",
                        enforce_detection=True, 
                        detector_backend="opencv" 
                    )
                    if embedding_objs:
                        embedding = embedding_objs[0]["embedding"]
                        known_face_db[person_name].append(embedding)
                        count += 1
                except Exception as e:
                    # Print exact error to debug "0 vectors" issue
                    logger.warning(f"Failed to load {filename}: {e}")
                
    logger.info(f"Loaded {count} face vectors for {len(known_face_db)} people.")

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
    best_iou = 0; best_id = None
    for track_id, body_box in body_boxes.items():
        iou = calculate_iou(face_box, body_box)
        if iou > 0.5 and iou > best_iou:
            best_iou = iou; best_id = track_id
    return best_id

def find_best_match_arcface(target_embedding):
    best_name = "Unknown"
    best_distance = 1.0
    # ArcFace Threshold: 0.65 is a good balance for distance
    ARCFACE_THRESHOLD = 0.65
    
    for name, embeddings_list in known_face_db.items():
        for known_emb in embeddings_list:
            try:
                a = np.array(target_embedding)
                b = np.array(known_emb)
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                distance = 1 - cos_sim
                
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
            except: pass
            
    return best_name, best_distance

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
            if output_frame is None: time.sleep(0.1); continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try: app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)
    except: pass

# --- RESOURCE LOADING ---
def _load_resources():
    global GFPGANer, GFPGAN_AVAILABLE, yolo_body_model, yolo_face_model, gfpgan_enhancer
    
    logger.info("Loading GFPGAN...")
    try:
        from gfpgan import GFPGANer as G; GFPGANer = G
        GFPGAN_AVAILABLE = True
    except: GFPGAN_AVAILABLE = False

    logger.info("Loading YOLO Body Model (GPU)...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR) 

    logger.info("Loading YOLO Face Model (GPU)...")
    face_path = get_face_model_path()
    if face_path:
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)

    if GFPGAN_AVAILABLE:
        logger.info("Loading GFPGAN Weights...")
        try:
            gfpgan_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE_STR)
        except: 
            with data_lock: server_data["face_enhancement_mode"] = "off_disabled"
            
    if DEEPFACE_AVAILABLE:
        logger.info("Initializing ArcFace Model...")
        DeepFace.build_model("ArcFace")

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
            cap.release(); time.sleep(0.5); continue
        with data_lock: latest_raw_frame = frame
    if cap: cap.release()

def video_processing_thread():
    global data_lock, output_frame, server_data, APP_SHOULD_QUIT, latest_raw_frame
    global person_registry, last_face_locations
    
    frame_count = 0
    # Fewer workers = less context switching for GPU
    executor = ThreadPoolExecutor(max_workers=4) 

    while not APP_SHOULD_QUIT:
        frame = None
        with data_lock:
            if latest_raw_frame is not None: frame = latest_raw_frame.copy()
        if frame is None: time.sleep(0.01); continue

        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        if frame is None: continue
        frame_count += 1
        
        with data_lock:
            enhancement_mode = server_data["face_enhancement_mode"]
            yolo_conf = server_data["yolo_conf"]

        # 1. YOLO Body Tracking (GPU)
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
                    person_registry[track_id] = {"name": "Unknown", "dist": 1.0, "last_seen": time.time()}

        # 2. YOLO Face Detection (GPU)
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model and DEEPFACE_AVAILABLE:
            face_results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
            
            current_face_locations = []
            if len(face_results) > 0:
                for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    current_face_locations.append((t, r, b, l)) 
            
            last_face_locations = current_face_locations

            # 3. Recognition (ArcFace)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for face_loc in current_face_locations:
                body_id = get_best_matching_body(face_loc, body_boxes)
                
                if body_id is not None:
                    top, right, bottom, left = face_loc
                    
                    # Pad
                    t_pad = max(0, top - 20); b_pad = min(frame.shape[0], bottom + 20)
                    l_pad = max(0, left - 20); r_pad = min(frame.shape[1], right + 20)
                    face_crop = rgb_frame[t_pad:b_pad, l_pad:r_pad]
                    
                    # GFPGAN
                    if GFPGAN_AVAILABLE and enhancement_mode == "on" and gfpgan_enhancer:
                         try:
                             crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                             _, _, restored_face = gfpgan_enhancer.enhance(crop_bgr, has_aligned=False, only_center_face=True, paste_back=False)
                             if restored_face is not None:
                                 face_crop = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
                         except: pass 

                    # ArcFace Embedding
                    try:
                        embedding_objs = DeepFace.represent(
                            img_path=face_crop,
                            model_name="ArcFace",
                            enforce_detection=False,
                            detector_backend="skip" # We already cropped it with YOLO
                        )
                        target_emb = embedding_objs[0]["embedding"]
                        name, distance = find_best_match_arcface(target_emb)
                        
                        # Smart Update
                        curr_data = person_registry[body_id]
                        curr_name = curr_data["name"]
                        curr_dist = curr_data["dist"]
                        update = False
                        
                        # 0.65 threshold
                        if distance < 0.65: 
                            if curr_name == "Unknown": update = True
                            elif name == curr_name and distance < curr_dist: update = True
                            elif name != curr_name and distance < (curr_dist - 0.15): update = True
                        
                        if update:
                            person_registry[body_id]["name"] = name
                            person_registry[body_id]["dist"] = distance
                        person_registry[body_id]["last_seen"] = time.time()
                    except: pass

        # 4. Drawing
        for (ft, fr, fb, fl) in last_face_locations:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

        live_face_payload = []
        for track_id in active_track_ids:
            t, r, b, l = body_boxes[track_id]
            data = person_registry.get(track_id, {"name": "Unknown", "dist": 1.0})
            name = data["name"]
            dist = data["dist"]
            conf_display = max(0, min(100, int((1.0 - dist) * 100)))
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = f"{name}"
            if name != "Unknown": label += f" ({conf_display}%)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
            cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)
            live_face_payload.append({"name": name, "confidence": conf_display})

        with data_lock:
            output_frame = frame
            server_data["live_faces"] = live_face_payload

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" HEADLESS VISION ENGINE STARTING ")
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