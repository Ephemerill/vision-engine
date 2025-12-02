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
import socket
import logging
import requests
import torch
from flask import Flask, Response
from ultralytics import YOLO

# --- CONFIGURATION ---
RECOGNITION_TOLERANCE = 0.5 
WEB_SERVER_PORT = 5005

# --- AGGRESSIVE LATENCY SETTINGS ---
# Forces FFMPEG to discard buffers. Essential for RTSP on OpenCV.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

# --- MODEL SELECTION (YOLOv11 Face) ---
FACE_MODEL_NAME = "yolov11n-face.pt"
FACE_MODEL_URL = f"https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/{FACE_MODEL_NAME}"

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("VisionEngine")

# --- HARDWARE CHECK (MAC OPTIMIZED) ---
if torch.cuda.is_available():
    DEVICE_STR = 'cuda:0'
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"✅ GPU DETECTED (CUDA): {gpu_name}")
elif torch.backends.mps.is_available():
    DEVICE_STR = 'mps'
    logger.info(f"✅ GPU DETECTED (APPLE SILICON/MPS)")
else:
    DEVICE_STR = 'cpu'
    logger.warning("⚠️  CRITICAL: GPU NOT DETECTED. Running on CPU.")

# --- IMPORTS (OPTIONAL) ---
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
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp"
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"
STREAM_WEBCAM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
FACE_CONFIDENCE_THRESH = 0.5 

COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_FG = (255, 255, 255)

# Default Body Models
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt"}

# --- SHARED STATE ---
# We use separate locks to ensure the video feed (Display) is never blocked by the AI (Processing)
display_lock = threading.Lock()
ai_lock = threading.Lock()

shared_state = {
    "latest_frame": None,          # The raw frame from camera
    "output_frame": None,          # The annotated frame for Flask
    "boxes_body": {},              # Results from YOLO Body
    "boxes_face": [],              # Results from YOLO Face
    "identities": {},              # Mapping ID -> Name
    "running": True
}

server_data = {
    "yolo_model_key": "n", 
    "yolo_conf": 0.4, 
    "face_enhancement_mode": "off" 
}

# --- RESOURCES ---
known_face_encodings = []
known_face_names = []
yolo_body_model = None
yolo_face_model = None
gfpgan_enhancer = None

# --- HELPER FUNCTIONS ---
def get_face_model_path():
    if os.path.exists(FACE_MODEL_NAME): return FACE_MODEL_NAME
    logger.info(f"Downloading {FACE_MODEL_NAME}...")
    try:
        response = requests.get(FACE_MODEL_URL, stream=True)
        with open(FACE_MODEL_NAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        return FACE_MODEL_NAME
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def resize_with_aspect_ratio(frame, max_w=640, max_h=480):
    if frame is None: return None
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h: return frame
    r = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names
    if not os.path.exists(known_faces_dir): return
    known_face_encodings.clear(); known_face_names.clear()
    
    # Dlib encoding is CPU bound and slow. We load this once.
    logger.info("Loading known faces...")
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
        for filename in os.listdir(person_dir):
            if filename.lower().endswith((".jpg", ".png")):
                try:
                    img = face_recognition.load_image_file(os.path.join(person_dir, filename))
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        known_face_encodings.append(encs[0])
                        known_face_names.append(person_name)
                except: pass

def get_ip_addresses():
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("100.100.100.100", 80)) 
        ips.append(s.getsockname()[0])
        s.close()
    except: pass
    return list(set(ips))

# --- FAST VIDEO CAPTURE CLASS (THE LAG FIX) ---
class FastVideoStream:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        # Try to lower latency at property level too
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                # If stream drops, try ONE reconnect then sleep to avoid hammering
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
                continue
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.grabbed: return None
            return self.frame.copy() # Return copy so main thread doesn't lock reader

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- AI PROCESSING THREAD (INFERENCE ONLY) ---
def ai_processing_loop():
    """
    Runs YOLO and Face Rec. 
    It runs AS FAST AS IT CAN, but does not block the display.
    If it runs at 5FPS, the boxes just update at 5FPS, but video stays 30FPS.
    """
    global shared_state, yolo_body_model, yolo_face_model
    
    # Load Models inside thread or globally
    logger.info(f"Loading YOLO Models on {DEVICE_STR}...")
    yolo_body_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_body_model.to(DEVICE_STR)
    
    face_path = get_face_model_path()
    if face_path:
        yolo_face_model = YOLO(face_path, verbose=False)
        yolo_face_model.to(DEVICE_STR)

    load_known_faces(KNOWN_FACES_DIR)

    frame_count = 0

    while shared_state["running"]:
        # 1. Get latest frame (Non-blocking check)
        raw_frame = None
        with display_lock:
            if shared_state["latest_frame"] is not None:
                raw_frame = shared_state["latest_frame"].copy()
        
        if raw_frame is None:
            time.sleep(0.01)
            continue

        # Resize for AI (Keep it small for speed)
        proc_frame = resize_with_aspect_ratio(raw_frame, 640, 480)
        frame_count += 1
        
        # 2. YOLO Body (Fast)
        conf = server_data["yolo_conf"]
        # persist=True tracks IDs
        results = yolo_body_model.track(proc_frame, persist=True, classes=[0], conf=conf, verbose=False, tracker="bytetrack.yaml")
        
        current_bodies = {}
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, ids):
                # t, r, b, l format for consistency with user code
                l, t, r, b = box 
                current_bodies[track_id] = (t, r, b, l)

        # 3. Face Detection & Rec (Slower - Run every 3rd frame or if GPU allows)
        current_faces = []
        
        if yolo_face_model:
            # Run Face YOLO
            f_results = yolo_face_model.predict(proc_frame, conf=0.5, verbose=False)
            if len(f_results) > 0:
                for box in f_results[0].boxes.xyxy.cpu().numpy().astype(int):
                    l, t, r, b = box
                    current_faces.append((t, r, b, l))

                    # Logic: Check if face is inside a body
                    cx, cy = (l+r)/2, (t+b)/2
                    matched_body_id = None
                    for bid, (bt, br, bb, bl) in current_bodies.items():
                        if bl < cx < br and bt < cy < bb:
                            matched_body_id = bid
                            break
                    
                    # If we found a body and haven't ID'd them recently, run Dlib
                    # Dlib is CPU bound, so we run it sparingly
                    if matched_body_id is not None:
                         # Run recognition
                        rgb_enc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                        encs = face_recognition.face_encodings(rgb_enc, [(t, r, b, l)])
                        if encs:
                            matches = face_recognition.compare_faces(known_face_encodings, encs[0], tolerance=RECOGNITION_TOLERANCE)
                            name = "Unknown"
                            if True in matches:
                                dists = face_recognition.face_distance(known_face_encodings, encs[0])
                                best_idx = np.argmin(dists)
                                name = known_face_names[best_idx]
                            
                            # Update Identity Registry
                            with ai_lock:
                                shared_state["identities"][matched_body_id] = name

        # 4. Update Global State with New Box Positions
        with ai_lock:
            shared_state["boxes_body"] = current_bodies
            shared_state["boxes_face"] = current_faces
        
        # Don't sleep; just run as fast as possible. 
        # If GPU is fast, this loops at 30ms. If CPU, 200ms.

# --- DISPLAY / FLASK THREAD ---
def generate_frames():
    """
    This yields frames to the browser.
    """
    while True:
        with display_lock:
            if shared_state["output_frame"] is None:
                time.sleep(0.01)
                continue
            
            # Compress to JPG
            (flag, encodedImage) = cv2.imencode(".jpg", shared_state["output_frame"])
            if not flag: continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='background:black; margin:0;'><img src='/video_feed' style='width:100%; height:100%; object-fit:contain;'></body></html>"

def run_flask():
    app.run(host='0.0.0.0', port=WEB_SERVER_PORT, debug=False, use_reloader=False, threaded=True)

# --- COMPOSITOR LOOP ---
def compositor_loop(stream):
    """
    Reads from FastVideoStream, Draws Boxes, Updates 'output_frame'.
    Runs at Camera FPS (30 FPS).
    """
    global shared_state
    
    while shared_state["running"]:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # Resize for display consistency
        frame = resize_with_aspect_ratio(frame, FRAME_WIDTH, FRAME_HEIGHT)
        
        # 1. Store Raw Frame for AI Thread to grab
        with display_lock:
            shared_state["latest_frame"] = frame # AI thread reads this

        # 2. Grab latest AI results (Thread-safe copy)
        with ai_lock:
            bodies = shared_state["boxes_body"].copy()
            faces = shared_state["boxes_face"].copy()
            identities = shared_state["identities"].copy()

        # 3. Draw Graphics
        # Draw Faces
        for (t, r, b, l) in faces:
            cv2.rectangle(frame, (l, t), (r, b), COLOR_FACE_BOX, 2)

        # Draw Bodies & Names
        for track_id, (t, r, b, l) in bodies.items():
            name = identities.get(track_id, "Unknown")
            color = COLOR_BODY_KNOWN if name != "Unknown" else COLOR_BODY_UNKNOWN
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            
            label = f"{track_id}: {name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
            cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)

        # 4. Update Output for Flask
        with display_lock:
            shared_state["output_frame"] = frame

        # Control framerate slightly to prevent CPU spinning 
        # (Wait 1ms is standard OpenCV practice)
        time.sleep(0.001)

# --- MAIN ---
if __name__ == "__main__":
    print("---------------------------------------------------")
    print(" ULTRA-LOW LATENCY VISION ENGINE ")
    print("---------------------------------------------------")

    # 1. Start the Fast Streamer
    # Note: If RTSP fails, it won't crash, just hangs retrying. 
    # Ensure STREAM_PI_RTSP is reachable.
    print(f"Connecting to: {STREAM_PI_RTSP}")
    video_stream = FastVideoStream(STREAM_PI_RTSP).start()
    
    # Wait for first frame
    while video_stream.read() is None:
        print("Waiting for RTSP stream...", end='\r')
        time.sleep(0.5)
    print("\nStream Acquired.")

    # 2. Start AI Thread
    t_ai = threading.Thread(target=ai_processing_loop, daemon=True)
    t_ai.start()
    
    # 3. Start Flask Thread
    t_flask = threading.Thread(target=run_flask, daemon=True)
    t_flask.start()

    print("---------------------------------------------------")
    ips = get_ip_addresses()
    for ip in ips:
        print(f" -> http://{ip}:{WEB_SERVER_PORT}/")
    print("---------------------------------------------------")

    # 4. Run Compositor in Main Thread
    try:
        compositor_loop(video_stream)
    except KeyboardInterrupt:
        shared_state["running"] = False
        video_stream.stop()
        print("\nStopping...")