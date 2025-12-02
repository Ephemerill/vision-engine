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
import traceback
import socket
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from flask import Flask, Response
from ultralytics import YOLO
from collections import deque

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
    try:
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU DETECTED: {gpu_name}")
    except:
        logger.info("✅ GPU detected (name unknown).")
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
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp"
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"
STREAM_WEBCAM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 
FACE_RECOGNITION_NTH_FRAME = 3 
TARGET_FPS = 20  # asked from ffmpeg (-r 20)

COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_FG = (255, 255, 255)

# Default Body Models (Standard YOLO11)
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt", "x": "yolo11x.pt"}

# --- GLOBAL STATE ---
data_lock = threading.Lock()
output_frame = None
latest_raw_frame = None 
latest_raw_frame_ts = 0.0
APP_SHOULD_QUIT = False
VIDEO_THREAD_STARTED = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

server_data = {
    "is_recording": False, "keyframe_count": 0, "action_result": "", "live_faces": [],
    "model": MODEL_GEMMA, 
    "yolo_model_key": "n", # Body model size
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

# Small helper to check ffmpeg presence
def ffmpeg_exists():
    try:
        subprocess.check_output(["ffmpeg", "-version"])
        return True
    except Exception:
        return False

# --- HELPER FUNCTIONS ---
def get_face_model_path():
    if os.path.exists(FACE_MODEL_NAME):
        return FACE_MODEL_NAME
        
    logger.info(f"Downloading New Face Model ({FACE_MODEL_NAME})...")
    try:
        response = requests.get(FACE_MODEL_URL, stream=True, timeout=30)
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
    if not os.path.exists(known_faces_dir): 
        logger.info("No known_faces directory found.")
        return
    
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
                    encs = face_recognition.face_encodings(img, model='small')
                    if encs:
                        known_face_encodings.append(encs[0])
                        known_face_names.append(person_name)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed encoding {image_path}: {e}")
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

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return "<html><body style='background:black; text-align:center;'><h1 style='color:white;'>Vision Engine Live</h1><img src='/video_feed' style='width:90%; border:2px solid #333;'></body></html>"

def generate_frames():
    global output_frame
    # Serve the latest frame; don't block waiting on old frames
    while not APP_SHOULD_QUIT:
        with data_lock:
            frame = output_frame.copy() if output_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        # Lower JPEG quality for faster encoding (change if you want better quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        (flag, encodedImage) = cv2.imencode(".jpg", frame, encode_param)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    # End generator on quit

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    try:
        app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)
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
    logger.info("Loading YOLO11 Body Model (GPU/CPU)...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    try:
        yolo_body_model.to(DEVICE_STR)
    except: pass

    # 2. Face Detection (YOLOv11-Face)
    face_path = get_face_model_path()
    if face_path:
        logger.info(f"Loading Face Model: {face_path} (GPU/CPU)...")
        yolo_face_model = YOLO(face_path, verbose=False)
        try:
            yolo_face_model.to(DEVICE_STR)
        except: pass
    else:
        logger.error("Failed to load Face Model.")

    # 3. Face Enhancement
    if GFPGAN_AVAILABLE:
        logger.info("Loading GFPGAN Weights...")
        try:
            gfpgan_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE_STR)
        except Exception as e:
            logger.warning(f"GFPGAN load failed: {e}")
            with data_lock: server_data["face_enhancement_mode"] = "off_disabled"

    load_known_faces(KNOWN_FACES_DIR)

# --- VIDEO READER: USE FFMPEG PIPE FOR LOW LATENCY ---
def _frame_reader_loop(source):
    """
    Aggressive low-latency reader using ffmpeg stdout rawvideo pipe.
    Falls back to cv2.VideoCapture if ffmpeg is not available or fails.
    """
    global latest_raw_frame, latest_raw_frame_ts, APP_SHOULD_QUIT, VIDEO_THREAD_STARTED, CURRENT_STREAM_SOURCE

    def cv_capture(src):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except:
                pass
        return cap

    # If source is an int or not an RTSP URL, use OpenCV path
    if isinstance(source, int) or (not source.startswith("rtsp://")) or (not ffmpeg_exists()):
        logger.info("Using OpenCV VideoCapture fallback.")
        cap = cv_capture(source)
        while not APP_SHOULD_QUIT:
            if cap is None or not cap.isOpened():
                cap = cv_capture(source)
                if not cap.isOpened():
                    time.sleep(1)
                    continue
                else:
                    VIDEO_THREAD_STARTED = True
                    logger.info("VideoCapture connected (fallback).")
            ret, frame = cap.read()
            if not ret:
                logger.warning("CV capture failed frame. reconnecting...")
                try:
                    cap.release()
                except:
                    pass
                cap = None
                time.sleep(0.2)
                continue
            with data_lock:
                latest_raw_frame = frame
                latest_raw_frame_ts = time.time()
        if cap:
            cap.release()
        return

    # Build ffmpeg command to output raw BGR24 frames scaled to target size
    ffmpeg_cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        "-i", source,
        "-an", "-sn",
        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-r", str(TARGET_FPS),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]

    logger.info("Starting ffmpeg reader for low-latency RTSP.")
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
        VIDEO_THREAD_STARTED = True
        frame_size = FRAME_WIDTH * FRAME_HEIGHT * 3
        while not APP_SHOULD_QUIT:
            # Read exact bytes for one frame
            raw = proc.stdout.read(frame_size)
            if not raw or len(raw) < frame_size:
                # EOF or truncated read -> try to restart ffmpeg
                logger.warning("FFmpeg frame read failed/truncated; restarting ffmpeg reader.")
                try:
                    proc.kill()
                except:
                    pass
                time.sleep(0.2)
                # attempt to restart
                proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
                continue
            # Convert to numpy image
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
            now = time.time()
            # Aggressively drop stale frames: only keep the newest frame
            with data_lock:
                latest_raw_frame = frame
                latest_raw_frame_ts = now
        # cleanup
        try:
            proc.kill()
        except:
            pass
    except Exception as e:
        logger.exception(f"FFmpeg reader failed: {e}")
        # fallback to cv2
        source_fallback = source
        logger.info("Falling back to OpenCV capture after ffmpeg failure.")
        cap = cv_capture(source_fallback)
        while not APP_SHOULD_QUIT:
            if cap is None or not cap.isOpened():
                cap = cv_capture(source_fallback)
                if not cap.isOpened():
                    time.sleep(1)
                    continue
                else:
                    VIDEO_THREAD_STARTED = True
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2); continue
            with data_lock:
                latest_raw_frame = frame
                latest_raw_frame_ts = time.time()
        if cap:
            cap.release()

# --- VIDEO PROCESSING THREAD (non-blocking, drop-stale-frame pattern) ---
def video_processing_thread():
    global data_lock, output_frame, server_data, APP_SHOULD_QUIT, latest_raw_frame
    global person_registry, last_face_locations, latest_raw_frame_ts
    
    frame_count = 0
    # limit workers appropriately: face encodings can be parallelized but don't create too many threads
    executor = ThreadPoolExecutor(max_workers=4)

    # keep a small local cache so we don't constantly re-lock server_data
    while not APP_SHOULD_QUIT:
        # Try to get the freshest frame. If another (newer) frame arrives while processing, we'll drop this one.
        with data_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None
            frame_ts = latest_raw_frame_ts

        if frame is None:
            time.sleep(0.005)
            continue

        # If the frame is older than 0.5s compared to now, skip it (drop)
        if time.time() - frame_ts > 0.5:
            # drop stale frame
            continue

        # Resize once to target (ffmpeg already scaled but keep safe)
        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        if frame is None:
            continue

        frame_count += 1

        # Read small copies of server config
        with data_lock:
            enhancement_mode = server_data["face_enhancement_mode"]
            yolo_conf = server_data["yolo_conf"]

        # 1. YOLO Body Tracking (YOLO11 - GPU if available)
        try:
            # Synchronous call -- track is stateful. Keep this synchronous but fast.
            body_results = yolo_body_model.track(frame, persist=True, classes=[0], conf=yolo_conf, verbose=False)
        except Exception as e:
            logger.warning(f"Body tracking failed: {e}")
            body_results = []

        body_boxes = {}; active_track_ids = []
        try:
            if len(body_results) > 0 and getattr(body_results[0].boxes, "id", None) is not None:
                boxes = body_results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = body_results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, track_ids):
                    l, t, r, b = box
                    # convert to (top, right, bottom, left)
                    body_boxes[int(track_id)] = (int(t), int(r), int(b), int(l))
                    active_track_ids.append(int(track_id))
                    if track_id not in person_registry:
                        person_registry[track_id] = {"name": "Unknown", "conf": 0.0, "last_seen": time.time()}
        except Exception as e:
            logger.debug(f"No body boxes: {e}")

        # 2. YOLO Face Detection (run only every Nth frame)
        current_face_locations = []
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model:
            try:
                face_results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
                if len(face_results) > 0:
                    for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                        l, t, r, b = box
                        current_face_locations.append((int(t), int(r), int(b), int(l)))
            except Exception as e:
                logger.warning(f"Face detection failed: {e}")

        # Publish last face locations for drawing
        last_face_locations = current_face_locations

        # 3. Recognition (offload encodings to executor)
        face_encoding_futures = []
        rgb_frame_for_encoding = None
        if current_face_locations and len(known_face_encodings) > 0:
            # convert once
            rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for face_loc in current_face_locations:
                # offload encoding to executor (fast model)
                future = executor.submit(face_recognition.face_encodings, rgb_frame_for_encoding, [face_loc], "small")
                face_encoding_futures.append((face_loc, future))

        # Process results of encodings (non-blocking wait with small timeout to avoid large stalls)
        for face_loc, future in face_encoding_futures:
            try:
                face_encs = future.result(timeout=0.45)  # keep timeout small so it won't block long
            except Exception as e:
                # encoding timed out or failed; skip this face
                logger.debug(f"Face encoding failed/timed out: {e}")
                continue

            if not face_encs:
                continue
            face_enc = face_encs[0]

            body_id = get_containing_body_box(face_loc, body_boxes)
            if body_id is None:
                continue
            name = "Unknown"; conf = 0.0
            try:
                matches = face_recognition.compare_faces(known_face_encodings, face_enc, tolerance=RECOGNITION_TOLERANCE)
                dists = face_recognition.face_distance(known_face_encodings, face_enc)
                if True in matches:
                    best_idx = np.argmin(dists)
                    name = known_face_names[best_idx]
                    conf = max(0, min(100, (1.0 - dists[best_idx]) * 100))
            except Exception as e:
                logger.debug(f"Face compare failed: {e}")

            if name != "Unknown":
                person_registry[body_id]["name"] = name
                person_registry[body_id]["conf"] = conf
            person_registry[body_id]["last_seen"] = time.time()

        # 4. Drawing overlay (fast)
        for (ft, fr, fb, fl) in last_face_locations:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

        live_face_payload = []
        for track_id in active_track_ids:
            t, r, b, l = body_boxes[track_id]
            data = person_registry.get(track_id, {"name": "Unknown", "conf": 0})
            name = data["name"]; conf = data["conf"]
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = f"{track_id}: {name}"
            if name != "Unknown": label += f" ({int(conf)}%)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
            cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)
            live_face_payload.append({"name": name, "confidence": conf})

        # Update output frame (only the freshest)
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