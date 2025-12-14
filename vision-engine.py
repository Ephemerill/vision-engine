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
    logger.warning("⚠️  CRITICAL: GPU NOT DETECTED. Running on CPU.")

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
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam" # Force TCP via code, not URL params
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"
STREAM_WEBCAM = 0

# --- VIDEO SOURCE SELECTION ---
# 0 = Webcam, 1 = RTSP
VIDEO_SOURCE_MODE = 0

if VIDEO_SOURCE_MODE == 1:
    CURRENT_STREAM_SOURCE = STREAM_PI_RTSP
    logger.info(f"Source: RTSP Stream ({STREAM_PI_RTSP})")
else:
    CURRENT_STREAM_SOURCE = STREAM_WEBCAM
    logger.info(f"Source: Webcam ({STREAM_WEBCAM})")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 
# FACE_RECOGNITION_NTH_FRAME is less relevant now as it runs on its own thread, but good for skipping
FACE_RECOGNITION_SKIP_FRAMES = 5 
BOX_PADDING = 10

COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_FG = (255, 255, 255)

# Default Body Models (Standard YOLO11)
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt", "x": "yolo11x.pt"}

# --- SHARED WORLD STATE ---
class WorldState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Latest raw frame from camera
        self.latest_frame = None
        self.latest_frame_ts = 0.0

        # Analysis Results
        self.body_boxes = {} # {track_id: (t, r, b, l)}
        self.face_locations = [] # [(t, r, b, l), ...]
        self.person_registry = {} # {track_id: {"name": str, "conf": float, "last_seen": ts}}
        
        # Display Only
        self.output_frame = None
        self.live_faces_payload = []
        
        # State Config
        self.is_running = True
        self.config = {
            "model": MODEL_GEMMA, 
            "yolo_model_key": "n", 
            "yolo_conf": 0.4, 
            "face_enhancement_mode": "off" 
        }

        if not GFPGAN_AVAILABLE:
            self.config["face_enhancement_mode"] = "off_disabled"

state = WorldState()

# --- RESOURCES ---
known_face_encodings = []
known_face_names = []

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
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80)) # Use Tailscale IP hint
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
    
    # Draw Background Box for readability
    cv2.rectangle(frame, (0, 0), (160, 120), (0, 0, 0), -1)
    
    # 1. Real-Time Lag
    lag_color = (0, 255, 0) if lag_ms < 200 else (0, 165, 255)
    if lag_ms > 1000: lag_color = (0, 0, 255)
    cv2.putText(frame, f"DISP LAG: {lag_ms:.0f} ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lag_color, 1)
    y += 18
    
    # 2. Individual Timings
    # cv2.putText(frame, f"Total Proc: {timings.get('total', 0):.1f}ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    # y += line_h
    # cv2.putText(frame, "----------------", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100,100,100), 1)
    # y += line_h
    
    keys = ["disp_draw", "body_inf", "face_inf", "recog"]
    for k in keys:
        if k in timings:
            val = timings[k]
            cv2.putText(frame, f"{k}: {val:.1f}ms", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200,200,200), 1)
            y += line_h
            
    return frame

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
    yolo_body_model = YOLO(YOLO_MODELS[state.config['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR) 

    # 2. Face Detection (YOLOv11-Face)
    face_path = get_face_model_path()
    if face_path:
        logger.info(f"Loading Face Model: {face_path} (GPU)...")
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)
    else:
        logger.error("Failed to load Face Model.")

    # 3. Face Enhancement
    if GFPGAN_AVAILABLE:
        logger.info("Loading GFPGAN Weights...")
        try:
            gfpgan_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE_STR)
        except: 
            with state.lock: state.config["face_enhancement_mode"] = "off_disabled"

    load_known_faces(KNOWN_FACES_DIR)

# --- THREAD 1: CAPTURE WORKER ---
def capture_thread(source):
    logger.info(f"Starting Speed Reader on: {source}")
    
    # CRITICAL: Set FFmpeg flags to ignore buffer.
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    cap = None
    while state.is_running:
        if cap is None or not cap.isOpened():
            if isinstance(source, int): cap = cv2.VideoCapture(source)
            else: cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            
            if cap.isOpened():
                logger.info("✅ Stream Connected (Low Latency Mode)")
            else:
                time.sleep(2); continue

        ret, frame = cap.read()
        if not ret: 
            logger.warning("Frame dropped. Reconnecting...")
            cap.release(); cap = None; time.sleep(0.5); continue
        
        # Pre-resize here to save bandwidth for all other threads
        frame_resized = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        
        with state.lock:
            state.latest_frame = frame_resized
            state.latest_frame_ts = time.time()
            
    if cap: cap.release()

# --- THREAD 2: BODY DETECTION WORKER ---
def body_worker():
    last_processed_ts = 0.0
    
    while state.is_running:
        # Get frame
        frame = None
        with state.lock:
            if state.latest_frame is not None and state.latest_frame_ts > last_processed_ts:
                frame = state.latest_frame.copy()
                last_processed_ts = state.latest_frame_ts
                conf = state.config["yolo_conf"]
        
        if frame is None or yolo_body_model is None:
            time.sleep(0.01); continue
            
        t0 = time.perf_counter()
        
        try:
            results = yolo_body_model.track(frame, persist=True, classes=[0], conf=conf, verbose=False)
        except Exception as e:
            logger.error(f"Body Track Error: {e}")
            time.sleep(0.1); continue
            
        new_body_boxes = {}
        new_active_ids = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                new_body_boxes[track_id] = (t, r, b, l)
                new_active_ids.append(track_id)
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        with state.lock:
            state.body_boxes = new_body_boxes
            # Initialize unknown people if new
            for tid in new_active_ids:
                if tid not in state.person_registry:
                    state.person_registry[tid] = {"name": "Unknown", "conf": 0.0, "last_seen": time.time()}
            # Store timing for debug
            if not hasattr(state, "perf_metrics"): state.perf_metrics = {}
            state.perf_metrics["body_inf"] = elapsed

# --- THREAD 3: FACE DETECTION WORKER ---
def face_worker():
    last_processed_ts = 0.0
    
    while state.is_running:
        frame = None
        with state.lock:
            if state.latest_frame is not None and state.latest_frame_ts > last_processed_ts:
                frame = state.latest_frame.copy()
                last_processed_ts = state.latest_frame_ts
        
        if frame is None or yolo_face_model is None:
            time.sleep(0.02); continue # Check slightly less often
            
        t0 = time.perf_counter()
        
        try:
            results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
        except Exception as e:
            logger.error(f"Face Det Error: {e}"); time.sleep(0.1); continue
            
        new_face_locs = []
        if len(results) > 0:
            for box in results[0].boxes.xyxy.cpu().numpy().astype(int):
                l, t, r, b = box
                new_face_locs.append((t, r, b, l))
                
        elapsed = (time.perf_counter() - t0) * 1000
        
        with state.lock:
            state.face_locations = new_face_locs
            if not hasattr(state, "perf_metrics"): state.perf_metrics = {}
            state.perf_metrics["face_inf"] = elapsed

# --- THREAD 4: RECOGNITION WORKER ---
def recognition_worker():
    # Runs less frequently
    frame_counter = 0
    
    while state.is_running:
        time.sleep(0.05) # Sleep a bit, we don't need 30fps recognition
        
        frame = None
        face_locs = []
        body_boxes = {}
        
        with state.lock:
            if state.latest_frame is None: continue
            frame = state.latest_frame.copy()
            face_locs = list(state.face_locations) # Copy
            body_boxes = dict(state.body_boxes) # Copy
            enhancement_mode = state.config["face_enhancement_mode"]

        if not face_locs: continue
        
        t0 = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Optimize: Only recognize faces that are inside bodies we are tracking
        # And maybe limit to 1-2 faces per pass to keep it fast
        
        for face_loc in face_locs:
            body_id = get_containing_body_box(face_loc, body_boxes)
            if body_id is not None:
                # Check if we already know this person with high confidence recently?
                # For now, just re-verify every time for robustness, but we could cache.
                
                top, right, bottom, left = face_loc
                input_image_encoding = rgb_frame
                encoding_loc = [face_loc]
                
                # GFPGAN
                if GFPGAN_AVAILABLE and enhancement_mode == "on" and gfpgan_enhancer:
                     t_pad = max(0, top - BOX_PADDING); b_pad = min(frame.shape[0], bottom + BOX_PADDING)
                     l_pad = max(0, left - BOX_PADDING); r_pad = min(frame.shape[1], right + BOX_PADDING)
                     face_crop_bgr = frame[t_pad:b_pad, l_pad:r_pad]
                     try:
                         _, _, restored_face = gfpgan_enhancer.enhance(
                             face_crop_bgr, has_aligned=False, only_center_face=True, paste_back=False
                         )
                         if restored_face is not None:
                             input_image_encoding = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
                             encoding_loc = [(0, input_image_encoding.shape[1], input_image_encoding.shape[0], 0)]
                     except: pass

                # Encode
                face_enc = face_recognition.face_encodings(input_image_encoding, encoding_loc)
                
                curr_name = "Unknown"
                curr_conf = 0.0
                
                if face_enc and len(known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(known_face_encodings, face_enc[0], tolerance=RECOGNITION_TOLERANCE)
                    dists = face_recognition.face_distance(known_face_encodings, face_enc[0])
                    if True in matches:
                        best_idx = np.argmin(dists)
                        curr_name = known_face_names[best_idx]
                        curr_conf = max(0, min(100, (1.0 - dists[best_idx]) * 100))
                
                # Update Registry
                with state.lock:
                    if body_id in state.person_registry:
                        if curr_name != "Unknown":
                            state.person_registry[body_id]["name"] = curr_name
                            state.person_registry[body_id]["conf"] = curr_conf
                        state.person_registry[body_id]["last_seen"] = time.time()
        
        elapsed = (time.perf_counter() - t0) * 1000
        with state.lock:
            if not hasattr(state, "perf_metrics"): state.perf_metrics = {}
            state.perf_metrics["recog"] = elapsed


# --- THREAD 5: DISPLAY / MAIN LOOP ---
def display_thread():
    while state.is_running:
        t_start = time.perf_counter()
        
        frame = None
        frame_ts = 0
        bodies = {}
        faces = []
        registry = {}
        perf = {}
        
        with state.lock:
            if state.latest_frame is not None:
                frame = state.latest_frame.copy()
                frame_ts = state.latest_frame_ts
            bodies = dict(state.body_boxes)
            faces = list(state.face_locations)
            registry = dict(state.person_registry)
            if hasattr(state, "perf_metrics"): perf = dict(state.perf_metrics)
        
        if frame is None:
            time.sleep(0.01); continue
            
        # Draw Faces
        for (ft, fr, fb, fl) in faces:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)
            
        # Draw Bodies & Identity
        live_faces_list = []
        for track_id, (bt, br, bb, bl) in bodies.items():
            # Get identity
            data = registry.get(track_id, {"name": "Unknown", "conf": 0.0})
            name = data["name"]
            conf = data["conf"]
            
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            cv2.rectangle(frame, (bl, bt), (br, bb), color, 2)
            
            label = f"{track_id}: {name}"
            if name != "Unknown": label += f" ({int(conf)}%)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (bl, bt - 25), (bl + tw + 10, bt), color, -1)
            cv2.putText(frame, label, (bl + 5, bt - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)
            
            live_faces_list.append({"name": name, "confidence": conf})
            
        # Calc Lag
        lag_ms = (time.time() - frame_ts) * 1000
        t_draw = (time.perf_counter() - t_start) * 1000
        perf["disp_draw"] = t_draw
        
        # Overlay
        frame = draw_perf_overlay(frame, perf, lag_ms)
        
        # Publish
        with state.lock:
            state.output_frame = frame
            state.live_faces_payload = live_faces_list
        
        # Cap at ~30 FPS for display loop to save CPU
        elapsed = time.perf_counter() - t_start
        sleep_time = max(0, (1/30.0) - elapsed)
        if sleep_time > 0: time.sleep(sleep_time)


# --- FLASK ROUTES ---
@app.route('/')
def index():
    return "<html><body style='background:black; text-align:center;'><h1 style='color:white;'>Vision Engine Multithreaded</h1><img src='/video_feed' style='width:90%; border:2px solid #333;'></body></html>"

def flask_gen():
    while True:
        with state.lock:
            if state.output_frame is None:
                time.sleep(0.02); continue
            (flag, encodedImage) = cv2.imencode(".jpg", state.output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(flask_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try:
        app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)
    except OSError:
        logger.error(f"Port {WEB_SERVER_PORT} in use.")

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" VISION ENGINE - MULTITHREADED ARCHITECTURE ")
    print("---------------------------------------------------")
    
    # 0. Load Resources Main Thread
    _load_resources()
    
    # 1. Capture Thread
    t_cap = threading.Thread(target=capture_thread, args=(CURRENT_STREAM_SOURCE,), daemon=True)
    t_cap.start()
    
    # 2. Body Thread
    t_body = threading.Thread(target=body_worker, daemon=True)
    t_body.start()
    
    # 3. Face Thread
    t_face = threading.Thread(target=face_worker, daemon=True)
    t_face.start()
    
    # 4. Recognition Thread
    t_rec = threading.Thread(target=recognition_worker, daemon=True)
    t_rec.start()
    
    # 5. Display Thread
    t_disp = threading.Thread(target=display_thread, daemon=True)
    t_disp.start()
    
    # 6. Flask Thread
    t_flask = threading.Thread(target=run_flask, daemon=True)
    t_flask.start()
    
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
            with state.lock:
                faces = state.live_faces_payload
                
            if faces:
                names = [f['name'] for f in faces]
                logger.info(f"Tracking: {names}")
            else:
                print(".", end="", flush=True)
    except KeyboardInterrupt:
        state.is_running = False
        print("\nShutting down...")