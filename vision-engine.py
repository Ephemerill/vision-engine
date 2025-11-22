import cv2
import face_recognition 
import os
import numpy as np
import ollama
import threading
import time
import sys
import subprocess 
import curses 
import base64 
from retinaface import RetinaFace
from ultralytics import YOLO
import traceback
from concurrent.futures import ThreadPoolExecutor
import torch

# --- CONFIGURATION (USER EDITABLE) ---
# ---------------------------------------------------------
# FACE RECOGNITION SENSITIVITY
# 0.6 = Standard (Can be loose)
# 0.5 = Stricter (Recommended for false positives)
# 0.4 = Very Strict (Might miss faces at angles)
RECOGNITION_TOLERANCE = 0.5 
# ---------------------------------------------------------

# --- DEVICE CONFIGURATION --- 
DEVICE_STR = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    DEVICE_STR = 'mps' 
TUI_INFO_MESSAGE = f"Using Device (YOLO/Torch): {DEVICE_STR}"

# --- SOTA IMPORTS ---
try:
    import deepface
    import gfpgan
    SOTA_AVAILABLE_PRECHECK = True
    GFPGAN_AVAILABLE_PRECHECK = True
except ImportError:
    SOTA_AVAILABLE_PRECHECK = False
    GFPGAN_AVAILABLE_PRECHECK = False

# --- STATUS & LOGGING ---
LOADING_STATUS_MESSAGE = ""
def print_progress_bar(step, total, message, bar_length=40):
    global LOADING_STATUS_MESSAGE
    percent = step / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    LOADING_STATUS_MESSAGE = f'[Step {step}/{total}] |{bar}| {int(percent*100)}% - {message}'

# --- AI MODELS ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_OFF = 'off'

# --- STREAM CONFIG ---
STREAM_PI_IP = "100.114.210.58"
STREAM_PI_RTSP = f"rtsp://admin:mysecretpassword@{STREAM_PI_IP}:8554/cam?rtsp_transport=tcp"
STREAM_PI_HLS = f"http://{STREAM_PI_IP}:8888/cam/index.m3u8"
STREAM_WEBCAM = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
ANALYSIS_COOLDOWN = 10
BOX_PADDING = 10

# --- ML THRESHOLDS ---
RETINAFACE_CONFIDENCE = 0.5 
FACE_RECOGNITION_NTH_FRAME = 5 

# --- COLORS ---
COLOR_BODY_KNOWN = (255, 100, 100) 
COLOR_BODY_UNKNOWN = (100, 100, 255) 
COLOR_FACE_BOX = (255, 255, 0) 
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT_FG = (255, 255, 255)

# --- YOLO MODELS ---
YOLO_MODELS = {"n": "yolo11n.pt", "s": "yolo11s.pt", "m": "yolo11m.pt"}

# --- GLOBAL STATE ---
data_lock = threading.Lock()
output_frame = None
latest_raw_frame = None 
APP_SHOULD_QUIT = False
SYSTEM_INITIALIZED = False
INITIALIZING = False
VIDEO_THREAD_STARTED = False
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP 

server_data = {
    "is_recording": False, "keyframe_count": 0, "action_result": "", "live_faces": [],
    "model": MODEL_GEMMA, 
    "yolo_model_key": "m", # Default
    "yolo_conf": 0.4, 
    "face_enhancement_mode": "off" 
}
if not GFPGAN_AVAILABLE_PRECHECK:
    server_data["face_enhancement_mode"] = "off_disabled"

# --- RESOURCES ---
known_face_encodings = []
known_face_names = []
person_registry = {} 
last_face_locations = [] 

DeepFace = None; dst = None; GFPGANer = None
yolo_model = None; gfpgan_enhancer = None
SOTA_AVAILABLE = False; GFPGAN_AVAILABLE = False

# --- HELPER FUNCTIONS ---
def resize_with_aspect_ratio(frame, max_w=640, max_h=480):
    if frame is None: return None
    h, w = frame.shape[:2]
    if w == 0 or h == 0: return frame
    if w <= max_w and h <= max_h: return frame
    r = min(max_w / w, max_h / h)
    return cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names, DeepFace, SOTA_AVAILABLE
    if not os.path.exists(known_faces_dir): return
    
    known_face_encodings.clear(); known_face_names.clear()
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
                except: pass

def get_containing_body_box(face_box, body_boxes):
    ft, fr, fb, fl = face_box
    cx, cy = (fl + fr) / 2, (ft + fb) / 2
    for track_id, (bt, br, bb, bl) in body_boxes.items():
        if bl < cx < br and bt < cy < bb: return track_id
    return None

# --- CONTROL FUNCTIONS ---
def set_yolo_model(model_key):
    global server_data, data_lock, yolo_model, LOADING_STATUS_MESSAGE
    if model_key not in YOLO_MODELS: return
    if model_key == server_data['yolo_model_key']: return
    
    def _loader():
        global yolo_model, LOADING_STATUS_MESSAGE
        with data_lock: LOADING_STATUS_MESSAGE = f"Switching YOLO to {model_key}..."
        try:
            new_model = YOLO(YOLO_MODELS[model_key], verbose=False)
            new_model.to('cpu') 
            # Atomic swap
            yolo_model = new_model
            with data_lock: 
                server_data['yolo_model_key'] = model_key
                LOADING_STATUS_MESSAGE = f"Switched to YOLO {model_key}"
                time.sleep(1); LOADING_STATUS_MESSAGE = ""
        except Exception as e:
            with data_lock: LOADING_STATUS_MESSAGE = f"YOLO Switch Error: {e}"

    threading.Thread(target=_loader, daemon=True).start()

def toggle_gfpgan():
    global server_data, data_lock, GFPGAN_AVAILABLE
    if not GFPGAN_AVAILABLE: return
    with data_lock:
        current = server_data['face_enhancement_mode']
        server_data['face_enhancement_mode'] = 'on' if current == 'off' else 'off'

# --- RESOURCE LOADING ---
def _load_resources():
    global DeepFace, dst, GFPGANer, SOTA_AVAILABLE, GFPGAN_AVAILABLE, yolo_model, gfpgan_enhancer
    
    print_progress_bar(1, 5, "Importing DeepFace...")
    try:
        from deepface import DeepFace as DF; DeepFace = DF
        from deepface.modules import verification as ver; dst = ver
        SOTA_AVAILABLE = True
    except: SOTA_AVAILABLE = False

    print_progress_bar(2, 5, "Importing GFPGAN...")
    try:
        from gfpgan import GFPGANer as G; GFPGANer = G
        GFPGAN_AVAILABLE = True
    except: GFPGAN_AVAILABLE = False

    print_progress_bar(3, 5, "Loading YOLO...")
    yolo_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_model.to('cpu') 

    print_progress_bar(4, 5, "Loading Enhancer...")
    if GFPGAN_AVAILABLE:
        try:
            gfpgan_enhancer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE_STR)
        except: 
            with data_lock: server_data["face_enhancement_mode"] = "off_disabled"

    print_progress_bar(5, 5, "Loading Faces...")
    load_known_faces(KNOWN_FACES_DIR)

# --- VIDEO THREADS ---
def _frame_reader_loop(source):
    global latest_raw_frame, data_lock, APP_SHOULD_QUIT, LOADING_STATUS_MESSAGE, VIDEO_THREAD_STARTED, CURRENT_STREAM_SOURCE
    cap = None
    
    def connect(src):
        if isinstance(src, int): return cv2.VideoCapture(src)
        return cv2.VideoCapture(src, cv2.CAP_FFMPEG) 

    while not APP_SHOULD_QUIT:
        if cap is None or not cap.isOpened():
            LOADING_STATUS_MESSAGE = f"Connecting to {source}..."
            cap = connect(source)
            if not cap.isOpened() and source == STREAM_PI_RTSP:
                source = STREAM_PI_HLS; CURRENT_STREAM_SOURCE = STREAM_PI_HLS; continue
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1); VIDEO_THREAD_STARTED = True; LOADING_STATUS_MESSAGE = ""
            else: time.sleep(2); continue

        ret, frame = cap.read()
        if not ret: cap.release(); time.sleep(0.5); continue
        with data_lock: latest_raw_frame = frame
    if cap: cap.release()

def video_processing_thread():
    global data_lock, output_frame, server_data, APP_SHOULD_QUIT, latest_raw_frame
    global person_registry, last_face_locations, LOADING_STATUS_MESSAGE
    
    frame_count = 0
    while INITIALIZING and not APP_SHOULD_QUIT: time.sleep(0.1)

    while not APP_SHOULD_QUIT:
        frame = None
        with data_lock:
            if latest_raw_frame is not None: frame = latest_raw_frame.copy()
        if frame is None: time.sleep(0.05); continue

        frame = cv2.flip(frame, 1)
        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        if frame is None: continue
        frame_count += 1
        
        # Get toggles
        with data_lock:
            enhancement_mode = server_data["face_enhancement_mode"]
            yolo_conf = server_data["yolo_conf"]

        # 1. YOLO Tracking
        yolo_results = yolo_model.track(frame, persist=True, classes=[0], conf=yolo_conf, verbose=False)
        body_boxes = {}; active_track_ids = []
        if yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                body_boxes[track_id] = (t, r, b, l)
                active_track_ids.append(track_id)
                if track_id not in person_registry:
                    person_registry[track_id] = {"name": "Unknown", "conf": 0.0, "last_seen": time.time()}

        # 2. RetinaFace Detection
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0:
            detected_faces = {}
            try: detected_faces = RetinaFace.detect_faces(frame, threshold=RETINAFACE_CONFIDENCE)
            except Exception as e: 
                with data_lock: LOADING_STATUS_MESSAGE = f"RetinaFace Error: {e}"

            current_face_locations = []
            if isinstance(detected_faces, dict):
                for key, data in detected_faces.items():
                    area = data['facial_area']
                    current_face_locations.append((int(area[1]), int(area[2]), int(area[3]), int(area[0])))
            last_face_locations = current_face_locations

            # 3. Recognition
            rgb_frame_for_encoding = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            for face_loc in current_face_locations:
                body_id = get_containing_body_box(face_loc, body_boxes)
                if body_id is not None:
                    # Crop
                    top, right, bottom, left = face_loc
                    
                    # --- GFPGAN ENHANCEMENT START ---
                    input_image_encoding = rgb_frame_for_encoding
                    
                    if GFPGAN_AVAILABLE and enhancement_mode == "on" and gfpgan_enhancer:
                         # Crop face strictly for enhancement
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
                         except:
                             encoding_loc = [face_loc] # Fallback
                    else:
                        encoding_loc = [face_loc]
                    # --- GFPGAN ENHANCEMENT END ---

                    face_enc = face_recognition.face_encodings(input_image_encoding, encoding_loc)
                    
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
                    person_registry[body_id]["last_seen"] = time.time()

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
            label = f"{track_id}: {name}"
            if name != "Unknown": label += f" ({int(conf)}%)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (l, t - 25), (l + tw + 10, t), color, -1)
            cv2.putText(frame, label, (l + 5, t - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_FG, 2)
            live_face_payload.append({"name": name, "confidence": conf})

        with data_lock:
            output_frame = frame
            server_data["live_faces"] = live_face_payload

# --- GUI / WINDOW MODE ---
def run_cv2_window_mode():
    """ Opens a desktop window for debugging with crash protection """
    global APP_SHOULD_QUIT
    print("Opening Debug Window... Press 'q' inside window to close.")
    window_name = "Debug Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while not APP_SHOULD_QUIT:
        frame_to_show = None
        with data_lock:
            if output_frame is not None:
                frame_to_show = output_frame.copy()
        
        if frame_to_show is not None:
            cv2.imshow(window_name, frame_to_show)
        else:
            cv2.imshow(window_name, np.zeros((480, 640, 3), dtype=np.uint8))

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        except: pass 

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
    
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    return "tui" 

# --- MAIN / TUI ---
def threaded_init():
    global INITIALIZING, SYSTEM_INITIALIZED
    INITIALIZING = True
    try:
        reader = threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True); reader.start()
        _load_resources()
        proc = threading.Thread(target=video_processing_thread, daemon=True); proc.start()
        SYSTEM_INITIALIZED = True
    except Exception as e:
        global LOADING_STATUS_MESSAGE
        LOADING_STATUS_MESSAGE = f"Init Error: {e}\n{traceback.format_exc()}"
    finally: INITIALIZING = False

def draw_tui(stdscr):
    global APP_SHOULD_QUIT
    curses.curs_set(0); stdscr.nodelay(True)
    curses.start_color(); curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    
    while not APP_SHOULD_QUIT:
        stdscr.clear()
        k = stdscr.getch()
        if k == ord('q'): APP_SHOULD_QUIT = True; return "quit"
        elif k == ord('i') and not SYSTEM_INITIALIZED and not INITIALIZING:
            threading.Thread(target=threaded_init, daemon=True).start()
        elif k == ord('o') and SYSTEM_INITIALIZED: return "open_window"
        elif k == ord('1'): set_yolo_model('n')
        elif k == ord('2'): set_yolo_model('s')
        elif k == ord('3'): set_yolo_model('m')
        elif k == ord('5'): toggle_gfpgan()
            
        if not SYSTEM_INITIALIZED:
            stdscr.addstr(0, 0, "Vision Engine (STOPPED)", curses.color_pair(1))
            stdscr.addstr(2, 0, "Press [i] to Initialize, [q] to Quit")
            stdscr.addstr(4, 0, f"Status: {LOADING_STATUS_MESSAGE}")
        else:
            with data_lock:
                 ym = server_data['yolo_model_key']
                 gm = server_data['face_enhancement_mode']
            
            stdscr.addstr(0, 0, "Vision Engine (RUNNING)", curses.color_pair(1))
            stdscr.addstr(1, 0, f"Source: {CURRENT_STREAM_SOURCE}")
            stdscr.addstr(2, 0, f"YOLO: {ym} (1-3) | GFPGAN: {gm} (5) | [o] Window | [q] Quit")
            
            with data_lock: faces = server_data["live_faces"]
            stdscr.addstr(4, 0, "--- TRACKED PEOPLE ---")
            for i, f in enumerate(faces):
                stdscr.addstr(5+i, 0, f"> {f['name']} ({int(f['confidence'])}%)")
            
            if LOADING_STATUS_MESSAGE:
                stdscr.addstr(10, 0, f"Log: {LOADING_STATUS_MESSAGE}")

        stdscr.refresh(); time.sleep(0.1)
    return "quit"

if __name__ == "__main__":
    next_action = "tui"
    try:
        while next_action != "quit":
            if next_action == "tui": next_action = curses.wrapper(draw_tui)
            elif next_action == "open_window": next_action = run_cv2_window_mode()
    except KeyboardInterrupt: pass
    finally:
        APP_SHOULD_QUIT = True
        print("Shutting down...")