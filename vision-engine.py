import warnings
warnings.filterwarnings("ignore")

import cv2
import face_recognition 
import os
import numpy as np
import threading
import time
import socket
import logging
import requests
import queue
import datetime
import torch
from flask import Flask, Response
from ultralytics import YOLO

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("VisionEngine")

# --- SETTINGS ---
STREAM_PI_IP = "100.114.210.58"
# We add aggressive flags directly to the URL just in case
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp&buffer_size=0"

WEB_SERVER_PORT = 5005
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FACE_MODEL_NAME = "yolov11n-face.pt"
FACE_MODEL_URL = f"https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/{FACE_MODEL_NAME}"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# --- GLOBAL STATE ---
results_lock = threading.Lock()
latest_results = {"body_boxes": {}, "active_ids": [], "face_boxes": []}
person_registry = {} 
recog_queue = queue.Queue(maxsize=1) 
APP_QUIT = False

# --- DIAGNOSTICS STATE ---
diag_stats = {
    "reader_fps": 0,
    "process_fps": 0,
    "last_log": time.time()
}

# --- FLASK ---
app = Flask(__name__)
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.ERROR)

def get_face_model_path():
    if os.path.exists(FACE_MODEL_NAME): return FACE_MODEL_NAME
    print(f"Downloading {FACE_MODEL_NAME}...")
    r = requests.get(FACE_MODEL_URL, stream=True)
    with open(FACE_MODEL_NAME, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    return FACE_MODEL_NAME

# --- AGGRESSIVE VIDEO READER ---
class DebugVideoReader:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.frame_ts = 0 # When we acquired the frame
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        
    def start(self):
        self.thread.start()
        return self

    def _run(self):
        # Force FFmpeg to have NO buffer.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|strict;experimental|analyzeduration;0|probesize;32"
        
        print(f"[READER] Connecting to {self.source}...")
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        
        while self.running:
            if not cap.isOpened():
                time.sleep(1)
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                continue

            # --- BUFFER DRAIN LOGIC ---
            # We measure how long 'read()' takes.
            # If it takes < 0.005s (5ms), it means the frame was ALREADY in memory (buffered).
            # We want to SKIP those until we hit a frame that makes us wait (Live Data).
            
            reads_performed = 0
            while True:
                t_start = time.time()
                ret, frame = cap.read()
                read_duration = time.time() - t_start
                reads_performed += 1
                
                if not ret:
                    print("[READER] Stream failed. Reconnecting.")
                    cap.release()
                    break

                # If read was instant, it's OLD data. Skip it.
                # Only accept frame if we waited > 5ms OR if we already skipped 5 bad ones.
                if read_duration > 0.005 or reads_performed > 5:
                    with self.lock:
                        self.frame = frame
                        self.frame_ts = time.time() # Mark the exact moment we got it
                        diag_stats["reader_fps"] += 1
                    break 
                
                # If we are here, we are discarding a buffered frame
                # print(f"Discarding buffered frame ({read_duration*1000:.2f}ms read)")

        cap.release()

    def get_latest(self):
        with self.lock:
            if self.frame is None: return None, 0
            return self.frame.copy(), self.frame_ts
            
    def stop(self):
        self.running = False
        self.thread.join()

stream = None

# --- FLASK ROUTE ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame', headers={'Cache-Control': 'no-cache, no-store, must-revalidate'})

def generate_frames():
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        if stream is None: time.sleep(0.1); continue
        
        # 1. Get Image
        frame, ts = stream.get_latest()
        if frame is None: time.sleep(0.01); continue
        
        # 2. Draw Visual Debug Time
        # This allows you to visually compare the video clock vs your wall clock
        now = datetime.datetime.now()
        time_str = now.strftime("%H:%M:%S.%f")[:-3]
        
        # Draw Overlays (Faces/Bodies)
        data = {}
        with results_lock: data = latest_results.copy()
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Draw Boxes
        for (t, r, b, l) in data.get("face_boxes", []):
            cv2.rectangle(frame, (l, t), (r, b), (255, 255, 0), 2)
            
        # Draw Time
        cv2.rectangle(frame, (0, 0), (640, 40), (0,0,0), -1)
        cv2.putText(frame, f"SYS TIME: {time_str}", (10, 30), font, 0.8, (0, 255, 0), 2)
        
        # Encode
        (flag, encodedImage) = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if flag:
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        
        time.sleep(0.01)

# --- WORKER THREADS ---
def recognition_worker():
    print("[WORKER] Recog thread started")
    known_encs, known_names = [], []
    
    # Load Known Faces
    if os.path.exists("known_faces"):
        for n in os.listdir("known_faces"):
            p = os.path.join("known_faces", n)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.endswith(".jpg"):
                        try:
                            img = face_recognition.load_image_file(os.path.join(p, f))
                            e = face_recognition.face_encodings(img)
                            if e: known_encs.append(e[0]); known_names.append(n)
                        except: pass
    
    while not APP_QUIT:
        try:
            job = recog_queue.get(timeout=0.1)
        except queue.Empty: continue
        
        # Process Recognition
        unknown_enc = face_recognition.face_encodings(job['frame'], [job['loc']])
        name, conf = "Unknown", 0.0
        
        if unknown_enc and known_encs:
            dists = face_recognition.face_distance(known_encs, unknown_enc[0])
            best = np.argmin(dists)
            if dists[best] < 0.5:
                name = known_names[best]
                conf = (1.0 - dists[best]) * 100
        
        if name != "Unknown":
            with threading.Lock(): # Update global registry
                 if job['id'] in person_registry:
                     person_registry[job['id']]['name'] = name

def processing_loop():
    yolo_body = YOLO("yolo11n.pt", verbose=False).to(DEVICE)
    yolo_face = None
    fp = get_face_model_path()
    if fp: yolo_face = YOLO(fp, verbose=False).to(DEVICE)
    
    print("[PROCESS] Models loaded. Loop starting.")
    
    while not APP_QUIT:
        if stream is None: time.sleep(1); continue
        
        frame, capture_ts = stream.get_latest()
        if frame is None: time.sleep(0.01); continue
        
        # --- LAG CHECK LOGGING ---
        # Calculate how old this frame is right NOW
        latency_ms = (time.time() - capture_ts) * 1000
        diag_stats["process_fps"] += 1
        
        # Log every 2 seconds
        if time.time() - diag_stats["last_log"] > 2.0:
            print(f"[LAG CHECK] Frame Age: {latency_ms:.1f}ms | Reader Rate: {diag_stats['reader_fps']/2:.1f}fps | Process Rate: {diag_stats['process_fps']/2:.1f}fps")
            diag_stats["reader_fps"] = 0
            diag_stats["process_fps"] = 0
            diag_stats["last_log"] = time.time()
            
            if latency_ms > 500:
                print("⚠️ CRITICAL LAG DETECTED: The Python Reader is holding old frames!")

        frame_sm = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Body Track
        res = yolo_body.track(frame_sm, persist=True, classes=[0], verbose=False)
        
        b_boxes = {}
        if res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = res[0].boxes.id.cpu().numpy().astype(int)
            for b, i in zip(boxes, ids):
                b_boxes[i] = (b[1], b[2], b[3], b[0]) # t, r, b, l
                if i not in person_registry: person_registry[i] = {"name": "Unknown"}

        # Face Detect (Every 3rd frame)
        f_boxes = []
        if yolo_face and (diag_stats["process_fps"] % 3 == 0):
            f_res = yolo_face.predict(frame_sm, conf=0.5, verbose=False)
            for b in f_res[0].boxes.xyxy.cpu().numpy().astype(int):
                f_boxes.append((b[1], b[2], b[3], b[0]))
                
                # Match to Body
                center_y = (b[1]+b[3])/2
                for bid, bbox in b_boxes.items():
                    if bbox[0] < center_y < bbox[2]: # Simple vertical check
                         if person_registry[bid]['name'] == "Unknown" and not recog_queue.full():
                             recog_queue.put({"id": bid, "frame": cv2.cvtColor(frame_sm, cv2.COLOR_BGR2RGB), "loc": (b[1], b[2], b[3], b[0])})

        with results_lock:
            latest_results["body_boxes"] = b_boxes
            latest_results["face_boxes"] = f_boxes

# --- MAIN ---
if __name__ == "__main__":
    print("--- DIAGNOSTIC MODE STARTING ---")
    stream = DebugVideoReader(STREAM_PI_RTSP).start()
    
    threading.Thread(target=processing_loop, daemon=True).start()
    threading.Thread(target=recognition_worker, daemon=True).start()
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False), daemon=True).start()

    ip = [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]
    print(f"\n✅ SERVER READY: http://{ip}:{WEB_SERVER_PORT}")
    print("👉 Look at the [LAG CHECK] logs in this terminal.")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        APP_QUIT = True
        stream.stop()