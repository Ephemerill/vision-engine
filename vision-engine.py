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
import av 
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
elif torch.backends.mps.is_available():
    DEVICE_STR = 'mps'
    logger.info("✅ APPLE SILICON (MPS) DETECTED")
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
STREAM_PI_IP = "100.114.210.58"
# TRY UDP FIRST (Lowest Latency), Fallback to TCP in code
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

# --- GLOBAL STATE ---
data_lock = threading.Lock()
output_frame = None

# (Frame, Capture Timestamp)
latest_raw_packet = (None, 0.0)

APP_SHOULD_QUIT = False
VIDEO_THREAD_STARTED = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

# Monitoring Metrics
metrics = {
    "transport": "INIT",
    "latency_pickup": 0,    # Time fram sat in memory
    "time_demux_gap": 0,    # Time between packets arriving (Jitter)
    "time_decode": 0,       # Time to decode packet to numpy
    "time_resize": 0,
    "time_body": 0,
    "time_face": 0,
    "time_recog": 0,
    "time_draw": 0,
    "time_total": 0,
    "skipped_frames": 0     # Frames dropped to catch up
}

server_data = {
    "is_recording": False, "keyframe_count": 0, "action_result": "", "live_faces": [],
    "model": "gemma3:4b", 
    "yolo_model_key": "n", 
    "yolo_conf": 0.4, 
    "face_enhancement_mode": "off" 
}

if not GFPGAN_AVAILABLE:
    server_data["face_enhancement_mode"] = "off_disabled"

# Default Body Models
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt", "l": "yolo11l.pt", "x": "yolo11x.pt"}

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
    logger.info(f"Downloading Face Model...")
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

def draw_metrics_overlay(frame, metrics):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    box_w, box_h = 240, 270
    cv2.rectangle(overlay, (w - box_w, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    x_start = w - box_w + 10
    y_start = 20
    line_h = 20
    
    def val_color(val, low=30, high=60):
        if val < low: return (0, 255, 0)
        if val < high: return (0, 255, 255)
        return (0, 0, 255)

    cv2.putText(frame, f"Transport: {metrics['transport']}", (x_start, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Packet Jitter (Are packets arriving smoothly?)
    jit = metrics['time_demux_gap']
    cv2.putText(frame, f"Pkt Jitter: {jit:.1f}ms", (x_start, y_start + line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color(jit, 40, 100), 1)

    # Decode Time
    dec = metrics['time_decode']
    cv2.putText(frame, f"Decoder: {dec:.1f}ms", (x_start, y_start + line_h*2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color(dec, 15, 30), 1)

    # Pickup Lag
    lag = metrics['latency_pickup']
    cv2.putText(frame, f"Pickup Lag: {lag:.1f}ms", (x_start, y_start + line_h*3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, val_color(lag, 50, 100), 1)

    # Skipped
    cv2.putText(frame, f"Drop/Skip: {metrics['skipped_frames']}", (x_start, y_start + line_h*4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.line(frame, (x_start, y_start + line_h*5), (w, y_start + line_h*5), (100,100,100), 1)

    # Processing Breakdown
    cv2.putText(frame, f"YOLO Body: {metrics['time_body']:.1f}ms", (x_start, y_start + line_h*6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"YOLO Face: {metrics['time_face']:.1f}ms", (x_start, y_start + line_h*7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"Recognize: {metrics['time_recog']:.1f}ms", (x_start, y_start + line_h*8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    tot = metrics['time_total']
    cv2.putText(frame, f"TOTAL: {tot:.1f}ms", (x_start, y_start + line_h*10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, val_color(tot, 50, 150), 2)
    return frame

# --- FLASK ---
@app.route('/')
def index(): return "<html><body style='background:black; text-align:center;'><img src='/video_feed' style='width:90%; border:2px solid #333;'></body></html>"

def generate_frames():
    while True:
        with data_lock:
            if output_frame is None: time.sleep(0.05); continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask(): app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False)

# --- RESOURCES ---
def _load_resources():
    global GFPGANer, GFPGAN_AVAILABLE, yolo_body_model, yolo_face_model, gfpgan_enhancer
    
    try: from gfpgan import GFPGANer as G; GFPGANer = G; GFPGAN_AVAILABLE = True
    except: GFPGAN_AVAILABLE = False

    logger.info("Loading YOLO11 Body Model...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR) 

    face_path = get_face_model_path()
    if face_path:
        logger.info(f"Loading Face Model...")
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)

    if GFPGAN_AVAILABLE:
        try: gfpgan_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE_STR)
        except: pass
    load_known_faces(KNOWN_FACES_DIR)

# --- LOW LATENCY READER ---
def _frame_reader_loop(source):
    global latest_raw_packet, data_lock, APP_SHOULD_QUIT, VIDEO_THREAD_STARTED, metrics
    
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        while not APP_SHOULD_QUIT:
            ret, frame = cap.read()
            if ret:
                with data_lock: latest_raw_packet = (frame, time.time())
            else: time.sleep(0.1)
        return

    # RETRY LOGIC for Protocol
    protocols = ['udp', 'tcp'] # Try UDP first for speed
    current_proto_idx = 0
    
    while not APP_SHOULD_QUIT:
        proto = protocols[current_proto_idx]
        metrics['transport'] = proto.upper()
        logger.info(f"Connecting via {proto.upper()}...")
        
        container = None
        try:
            container = av.open(source, options={
                'rtsp_transport': proto,
                'fflags': 'nobuffer',
                'flags': 'low_delay',
                'probesize': '32',
                'analyzeduration': '0',
                'max_delay': '500000', # 0.5s max buffer
                'reorder_queue_size': '0'
            })
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            
            logger.info(f"Connected ({proto}). Streaming...")
            VIDEO_THREAD_STARTED = True
            
            last_pkt_time = time.time()
            
            # DEMUX LOOP (Separate from Decode)
            for packet in container.demux(stream):
                if APP_SHOULD_QUIT: break
                
                # Jitter Monitor
                now = time.time()
                metrics['time_demux_gap'] = (now - last_pkt_time) * 1000
                last_pkt_time = now
                
                # Decode
                if packet.dts is None: continue
                
                # Monitor Decode Time
                t_dec_start = time.perf_counter()
                frames = packet.decode()
                
                for frame in frames:
                    # Convert to numpy
                    img = frame.to_ndarray(format='bgr24')
                    t_dec_end = time.perf_counter()
                    metrics['time_decode'] = (t_dec_end - t_dec_start) * 1000
                    
                    with data_lock:
                        latest_raw_packet = (img, time.time())
                
        except av.AVError as e:
            logger.error(f"Stream Error ({proto}): {e}")
            # Switch protocol on error
            current_proto_idx = (current_proto_idx + 1) % len(protocols)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Reader Crash: {e}")
            time.sleep(1)
        finally:
            if container: container.close()

# --- PROCESSOR ---
def video_processing_thread():
    global data_lock, output_frame, server_data, APP_SHOULD_QUIT, latest_raw_packet, metrics
    global person_registry, last_face_locations
    
    frame_count = 0
    while latest_raw_packet[0] is None and not APP_SHOULD_QUIT: time.sleep(0.1)

    while not APP_SHOULD_QUIT:
        t_start_loop = time.perf_counter()
        
        with data_lock:
            frame, capture_ts = latest_raw_packet
        
        if frame is None:
            time.sleep(0.001); continue

        # Pickup Lag
        metrics['latency_pickup'] = (time.time() - capture_ts) * 1000
        
        # 2. RESIZE
        t0 = time.perf_counter()
        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        metrics['time_resize'] = (time.perf_counter() - t0) * 1000
        if frame is None: continue
        frame_count += 1
        
        with data_lock:
            yolo_conf = server_data["yolo_conf"]

        # 3. YOLO BODY
        t0 = time.perf_counter()
        active_track_ids = []
        body_boxes = {}
        try:
            body_results = yolo_body_model.track(frame, persist=True, classes=[0], conf=yolo_conf, verbose=False)
            if body_results[0].boxes.id is not None:
                boxes = body_results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = body_results[0].boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(boxes, track_ids):
                    l, t, r, b = box
                    body_boxes[track_id] = (t, r, b, l)
                    active_track_ids.append(track_id)
                    if track_id not in person_registry:
                        person_registry[track_id] = {"name": "Unknown", "conf": 0.0, "last_seen": time.time()}
        except: pass
        metrics['time_body'] = (time.perf_counter() - t0) * 1000

        # 4. YOLO FACE
        t0 = time.perf_counter()
        did_face_detect = False
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0 and yolo_face_model:
            did_face_detect = True
            face_results = yolo_face_model.predict(frame, conf=FACE_CONFIDENCE_THRESH, verbose=False)
            current_face_locations = []
            if len(face_results) > 0:
                for box in face_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    current_face_locations.append((t, r, b, l))
            last_face_locations = current_face_locations
        metrics['time_face'] = (time.perf_counter() - t0) * 1000

        # 5. RECOGNITION
        t0 = time.perf_counter()
        if did_face_detect:
            rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for face_loc in last_face_locations:
                body_id = get_containing_body_box(face_loc, body_boxes)
                if body_id is not None:
                    encoding_loc = [face_loc]
                    face_enc = face_recognition.face_encodings(rgb_frame_for_encoding, encoding_loc)
                    
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
        metrics['time_recog'] = (time.perf_counter() - t0) * 1000

        # 6. DRAWING
        t0 = time.perf_counter()
        for (ft, fr, fb, fl) in last_face_locations:
            cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

        live_face_payload = []
        for track_id in active_track_ids:
            t, r, b, l = body_boxes[track_id]
            data = person_registry.get(track_id, {"name": "Unknown", "conf": 0})
            name = data["name"]; conf = data["conf"]
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"{track_id}:{name}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            live_face_payload.append({"name": name, "confidence": conf})
        metrics['time_draw'] = (time.perf_counter() - t0) * 1000
        
        metrics['time_total'] = (time.perf_counter() - t_start_loop) * 1000
        frame = draw_metrics_overlay(frame, metrics)

        with data_lock:
            output_frame = frame
            server_data["live_faces"] = live_face_payload

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" VISION ENGINE: LOW LATENCY MODE ")
    print("---------------------------------------------------")
    reader = threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True)
    reader.start()
    _load_resources()
    proc = threading.Thread(target=video_processing_thread, daemon=True)
    proc.start()
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f" -> http://{get_ip_addresses()[0]}:{WEB_SERVER_PORT}/")
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt: APP_SHOULD_QUIT = True