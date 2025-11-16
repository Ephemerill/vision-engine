import cv2
import face_recognition # Still used for Dlib fallback
import os
import numpy as np
import ollama
import threading
import time
import sys
import subprocess # For self-update
import curses # For TUI
import base64 # Re-added for ollama calls
from retinaface import RetinaFace
from ultralytics import YOLO

# --- NEW: Parallelism ---
from concurrent.futures import ThreadPoolExecutor
# --- NEW: Better Error Logging ---
import traceback

# --- GPU/Device Configuration ---
import torch
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
TUI_INFO_MESSAGE = f"Using Device: {DEVICE}" # For display in TUI
# -------------------------------------

# --- New SOTA Imports (with better error handling) ---
try:
    import deepface
    import gfpgan
    SOTA_AVAILABLE_PRECHECK = True
    GFPGAN_AVAILABLE_PRECHECK = True
except ImportError:
    SOTA_AVAILABLE_PRECHECK = False
    GFPGAN_AVAILABLE_PRECHECK = False
# ---------------------------

# --- ✨ ASCII Progress Bar Function ---
def print_progress_bar(step, total, message, bar_length=40):
    """MODIFIED: Updates a global status var instead of printing."""
    global LOADING_STATUS_MESSAGE
    percent = step / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    # Update the global message for the TUI to draw
    LOADING_STATUS_MESSAGE = f'[Step {step}/{total}] |{bar}| {int(percent*100)}% - {message}'
    if step == total:
        LOADING_STATUS_MESSAGE = f'[Step {step}/{total}] |{bar}| 100% - {message}'
        LOADING_STATUS_MESSAGE += "\nAll models and faces loaded!"
# ------------------------------------

# --- Configuration ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_LLAVA = 'llava'
MODEL_MOONDREAM = 'moondream'
MODEL_OFF = 'off'

# --- Video Source Configuration ---
STREAM_WEBCAM = 0 # Default built-in webcam

# --- <<< IP ADDRESS CHECK >>> ---
# Please double-check this is your Pi's correct Tailscale IP.
STREAM_PI_TAILSCALE_IP = "100.114.210.58" 
STREAM_PI_RTSP = f"rtsp://{STREAM_PI_TAILSCALE_IP}:8554/cam"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
KNOWN_FACES_DIR = "known_faces"
ANALYSIS_COOLDOWN = 10
BOX_PADDING = 10

# --- SOTA Face Recognition Config ---
SOTA_MODEL = "Dlib"
SOTA_METRIC = "euclidean"
MAX_RECOGNITION_DISTANCE = 0.6
if SOTA_AVAILABLE_PRECHECK:
    SOTA_MODEL = "ArcFace"
    SOTA_METRIC = "cosine"
    
DEFAULT_RECOGNITION_THRESHOLD = MAX_RECOGNITION_DISTANCE 
DEFAULT_RETINAFACE_CONF = 0.9 
ACTION_MOTION_THRESHOLD = 30
FACE_RECOGNITION_NTH_FRAME = 5 

# --- Drawing Colors ---
COLOR_BODY_KNOWN = (255, 100, 100) # Light Blue
COLOR_BODY_UNKNOWN = (100, 100, 255) # Light Red
COLOR_FACE_BOX = (0, 255, 255) # Yellow
COLOR_TEXT_BACKGROUND = (0, 0, 0) # Black
COLOR_TEXT_FOREGROUND = (255, 255, 255) # White
# ---------------------

# --- YOLO Model Definitions ---
YOLO_MODELS = {
    "n": "yolo11n.pt", # Nano (Fastest)
    "s": "yolo11s.pt", # Small (Balanced)
    "m": "yolo11m.pt",  # Medium (Accurate)
    "x": "yolo11x.pt"   # Ultra
}

# --- Global Server State ---
data_lock = threading.Lock()
output_frame = None
latest_raw_frame = None # For the reader thread

# --- TUI STATE ---
APP_SHOULD_QUIT = False
SYSTEM_INITIALIZED = False
INITIALIZING = False
VIDEO_THREAD_STARTED = False
LOADING_STATUS_MESSAGE = ""
video_thread = None
reader_thread = None
CURRENT_STREAM_SOURCE = STREAM_PI_RTSP # Default to Pi Stream
# -------------------

server_data = {
    "is_recording": False,
    "keyframe_count": 0,
    "action_result": "",
    "live_faces": [],
    "model": MODEL_GEMMA, 
    "yolo_model_key": "m",
    "yolo_conf": 0.4,
    "yolo_imgsz": 640,
    "retinaface_conf": DEFAULT_RETINAFACE_CONF,
    "recognition_threshold": DEFAULT_RECOGNITION_THRESHOLD,
    "face_alignment_mode": "fast", 
    "face_enhancement_mode": "off"
}
if not GFPGAN_AVAILABLE_PRECHECK:
    server_data["face_enhancement_mode"] = "off_disabled" 

# --- Global ML State (Loaded on init) ---
known_face_encodings = []
known_face_names = []
analysis_results = {}
action_thread = None
stop_action_thread = False
action_frames = []
last_action_frame_gray = None

# Models are None until load_resources() is called
yolo_model = None 
gfpgan_enhancer = None 
person_registry = {} 

# --- Heavy Imports (will be called by load_resources) ---
DeepFace = None
dst = None
GFPGANer = None
SOTA_AVAILABLE = False
GFPGAN_AVAILABLE = False

# --- NEW: Aspect Ratio Resize Helper ---
def resize_with_aspect_ratio(frame, max_w=640, max_h=480):
    """
    Resizes an image to fit within max_w and max_h, preserving
    its aspect ratio.
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame
    if w <= max_w and h <= max_h:
        return frame
    r = min(max_w / w, max_h / h)
    new_w = int(w * r)
    new_h = int(h * r)
    if new_w == 0 or new_h == 0:
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

# --- Helper Functions (AI threads, etc.) ---

def action_comprehension_thread():
    global data_lock, output_frame, server_data, stop_action_thread, last_action_frame_gray
    chat_messages = []
    last_frame_gray = None
    keyframe_count = 0
    with data_lock:
        last_frame_gray = last_action_frame_gray 

    while not stop_action_thread:
        current_frame = None
        with data_lock:
            if output_frame is not None:
                current_frame = output_frame.copy()
            current_model = server_data['model']
        if current_frame is None:
            time.sleep(0.1); continue
            
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
        is_keyframe = False
        if last_frame_gray is None:
            is_keyframe = True
        else:
            frame_delta = cv2.absdiff(last_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            if thresh.sum() > 0: is_keyframe = True
            
        if is_keyframe:
            last_frame_gray = gray; keyframe_count += 1
            with data_lock: server_data["keyframe_count"] = keyframe_count
            
            success, buffer = cv2.imencode('.jpg', current_frame)
            if not success: continue
            b64_frame = base64.b64encode(buffer).decode('utf-8')
            
            prompt = "";
            if current_model == MODEL_MOONDREAM:
                prompt = "What action is happening in this image?"
            else:
                if not chat_messages: prompt = "This is the first keyframe. Briefly describe what is happening."
                else: prompt = "This is the next keyframe. Briefly describe the new action."
            
            chat_messages.append({"role": "user", "content": prompt, "images": [b64_frame]})
            try:
                response = ollama.chat(model=current_model, messages=chat_messages, stream=False)
                response_message = response['message']; response_content = response_message['content']
                chat_messages.append(response_message)
                with data_lock: server_data["action_result"] += f"- {response_content}\n"
            except Exception as e:
                with data_lock: server_data["action_result"] = "Error connecting to Ollama."
                time.sleep(2)
        time.sleep(0.5) 

def analyze_frame_with_gemma(frame, name):
    global analysis_results, data_lock
    success, buffer = cv2.imencode('.jpg', frame)
    if not success: 
        with data_lock: analysis_results[name] = "Error: Failed to encode frame."
        return
    b64_image = base64.b64encode(buffer).decode('utf-8')
    with data_lock: current_model = server_data['model']
    if current_model == MODEL_OFF:
        return
    prompt = ""
    if current_model == MODEL_MOONDREAM:
        prompt = "What is the person in this image doing?"
    else:
        prompt = f"This person has been identified as {name}. Briefly describe what they are doing."
    try:
        response = ollama.chat(model=current_model, messages=[{'role': 'user', 'content': prompt, 'images': [b64_image]}], stream=False)
        analysis_text = response['message']['content']
        with data_lock: analysis_results[name] = analysis_text
    except Exception as e:
        with data_lock: analysis_results[name] = "Error connecting to Ollama."

# --- SOTA-Aware Face Loader ---
def load_known_faces(known_faces_dir):
    global known_face_encodings, known_face_names, SOTA_MODEL, SOTA_METRIC, MAX_RECOGNITION_DISTANCE, DEFAULT_RECOGNITION_THRESHOLD
    global DeepFace, dst
    
    if SOTA_MODEL != "Dlib":
        if not os.path.exists(known_faces_dir): return
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        representation = DeepFace.represent(
                            img_path=image_path, 
                            model_name=SOTA_MODEL,
                            enforce_detection=True,
                            detector_backend='retinaface' 
                        )
                        if representation and len(representation) > 0:
                            known_face_encodings.append(representation[0]["embedding"])
                            known_face_names.append(person_name)
                    except Exception as e:
                        pass 
        if not known_face_encodings:
            SOTA_MODEL = "Dlib"
            SOTA_METRIC = "euclidean"
            MAX_RECOGNITION_DISTANCE = 0.6
            DEFAULT_RECOGNITION_THRESHOLD = 0.6
            with data_lock:
                 server_data["recognition_threshold"] = DEFAULT_RECOGNITION_THRESHOLD

    if SOTA_MODEL == "Dlib":
        if not os.path.exists(known_faces_dir): return
        known_face_encodings.clear()
        known_face_names.clear()
        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encodings_dlib = face_recognition.face_encodings(face_image)
                        for encoding in face_encodings_dlib:
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                    except Exception as e:
                        pass 

def get_containing_body_box(face_box, body_boxes):
    ft, fr, fb, fl = face_box
    face_center_x = (fl + fr) / 2
    face_center_y = (ft + fb) / 2
    for track_id, (bt, br, bb, bl) in body_boxes.items():
        if bl < face_center_x < br and bt < face_center_y < bb:
            return track_id
    return None

# --- <<< MODIFIED: Sequential, Verbose Resource Loading >>> ---
# We've broken the parallel loader into sequential, logged steps
# to easily identify hangs and errors.

def _load_imports():
    global DeepFace, dst, GFPGANer, SOTA_AVAILABLE, GFPGAN_AVAILABLE
    print_progress_bar(1, 6, "Importing heavy libraries...")
    try:
        from deepface import DeepFace as DF_Import
        DeepFace = DF_Import
        from deepface.modules import verification as dst_Import
        dst = dst_Import
        SOTA_AVAILABLE = True
    except ImportError:
        SOTA_AVAILABLE = False
    
    try:
        from gfpgan import GFPGANer as G_Import
        GFPGANer = G_Import
        GFPGAN_AVAILABLE = True
    except ImportError:
        GFPGAN_AVAILABLE = False
        with data_lock:
            server_data["face_enhancement_mode"] = "off_disabled"

def _load_deepface():
    global SOTA_MODEL, SOTA_METRIC, MAX_RECOGNITION_DISTANCE, DEFAULT_RECOGNITION_THRESHOLD
    print_progress_bar(2, 6, "Loading SOTA Model (ArcFace)...")
    if SOTA_AVAILABLE:
        try:
            DeepFace.build_model(SOTA_MODEL)
            MAX_RECOGNITION_DISTANCE = dst.find_threshold(SOTA_MODEL, SOTA_METRIC) 
        except Exception as e:
            SOTA_MODEL = "Dlib"; SOTA_METRIC = "euclidean"; MAX_RECOGNITION_DISTANCE = 0.6
    else:
        SOTA_MODEL = "Dlib"; SOTA_METRIC = "euclidean"; MAX_RECOGNITION_DISTANCE = 0.6
    DEFAULT_RECOGNITION_THRESHOLD = MAX_RECOGNITION_DISTANCE
    with data_lock:
        server_data["recognition_threshold"] = DEFAULT_RECOGNITION_THRESHOLD

def _load_yolo():
    global yolo_model
    print_progress_bar(3, 6, f"Loading YOLO Model to {DEVICE}...")
    yolo_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)
    yolo_model.to(DEVICE)

def _load_gfpgan():
    global gfpgan_enhancer
    print_progress_bar(4, 6, f"Loading GFPGAN Model to {DEVICE}...")
    if GFPGAN_AVAILABLE:
        try:
            gfpgan_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=DEVICE
            )
        except Exception as e:
            with data_lock: server_data["face_enhancement_mode"] = "off_disabled"

def _load_faces():
    print_progress_bar(5, 6, "Loading Known Faces...")
    load_known_faces(KNOWN_FACES_DIR)

# --- <<< END of new loading functions >>> ---


# --- <<< NEW: FRAME READER (PRODUCER) THREAD >>> ---
def _frame_reader_loop(source):
    """
    This is the "Reader" thread.
    Its only job is to read frames from the source and update latest_raw_frame.
    """
    global latest_raw_frame, data_lock, APP_SHOULD_QUIT, LOADING_STATUS_MESSAGE, VIDEO_THREAD_STARTED
    
    cap = None
    try:
        # --- MODIFIED: Set a 5-second timeout on the connection ---
        # This will prevent the 30-second hang if the IP is wrong
        # We use a custom environment variable for OpenCV
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp;stimeout;5000000'
        LOADING_STATUS_MESSAGE = f"Attempting to connect to stream: {source}"
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            LOADING_STATUS_MESSAGE = f"Error: Could not open video source {source}. Check IP/firewall."
            VIDEO_THREAD_STARTED = False # Graceful failure
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        VIDEO_THREAD_STARTED = True # Mark as started *only after* cap is open
        LOADING_STATUS_MESSAGE = "Video stream connected."
        
    except Exception as e:
        LOADING_STATUS_MESSAGE = f"Error opening source: {e}\n{traceback.format_exc()}"
        VIDEO_THREAD_STARTED = False # Graceful failure
        return

    while not APP_SHOULD_QUIT:
        try:
            ret, frame = cap.read()
            if not ret:
                LOADING_STATUS_MESSAGE = "Error: Cannot read frame. Reconnecting..."
                cap.release()
                cap = cv2.VideoCapture(source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(0.5)
                continue
            
            with data_lock:
                global latest_raw_frame
                latest_raw_frame = frame
                
        except Exception as e:
            LOADING_STATUS_MESSAGE = f"Stream read error: {e}"
            time.sleep(0.1)
    
    if cap:
        cap.release()
    LOADING_STATUS_MESSAGE = "Reader thread stopped."
    VIDEO_THREAD_STARTED = False # Graceful failure


# --- <<< MODIFIED: VIDEO PROCESSOR (CONSUMER) THREAD >>> ---
def video_processing_thread(): 
    """
    This is the "Processor" thread.
    It reads from latest_raw_frame, processes, and updates output_frame.
    """
    global data_lock, output_frame, server_data, APP_SHOULD_QUIT, latest_raw_frame
    global analysis_results, yolo_model, person_registry, gfpgan_enhancer
    global DeepFace, dst, SOTA_AVAILABLE, GFPGAN_AVAILABLE, DEVICE
    global LOADING_STATUS_MESSAGE, VIDEO_THREAD_STARTED
    
    frame_count = 0
    last_analysis_time = {} 

    # --- <<< NEW: Helper function for parallel face processing >>> ---
    def _process_single_face(face_data_in):
        """
        Takes one face, runs all ML, returns result.
        This runs in a thread pool.
        """
        rgb_frame, face_location, current_alignment_mode, current_enhancement_mode = face_data_in
        name, confidence = "Unknown", 0
        current_face_encoding = None

        try:
            if SOTA_MODEL != "Dlib":
                t, r, b, l = face_location
                t_pad = max(0, t - BOX_PADDING); b_pad = min(rgb_frame.shape[0], b + BOX_PADDING)
                l_pad = max(0, l - BOX_PADDING); r_pad = min(rgb_frame.shape[1], r + BOX_PADDING)
                face_crop_rgb = rgb_frame[t_pad:b_pad, l_pad:r_pad]
                if face_crop_rgb.size == 0: return (face_location, "Unknown", 0)
                
                input_face = face_crop_rgb 
                if GFPGAN_AVAILABLE and current_enhancement_mode == "on" and gfpgan_enhancer is not None:
                    try:
                        face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
                        _, _, restored_face_image = gfpgan_enhancer.enhance(
                            face_crop_bgr, has_aligned=False, only_center_face=True, paste_back=True
                        )
                        if restored_face_image is not None:
                            input_face = cv2.cvtColor(restored_face_image, cv2.COLOR_BGR2RGB)
                    except Exception as e: pass 

                if SOTA_AVAILABLE:
                    if current_alignment_mode == "accurate":
                        representation = DeepFace.represent(
                            img_path=input_face, model_name=SOTA_MODEL,
                            enforce_detection=True, detector_backend='retinaface'
                        )
                    else:
                        representation = DeepFace.represent(
                            img_path=input_face, model_name=SOTA_MODEL,
                            enforce_detection=False, detector_backend='skip'
                        )
                    current_face_encoding = representation[0]["embedding"]
            else:
                encodings = face_recognition.face_encodings(rgb_frame, [face_location])
                if encodings:
                    current_face_encoding = encodings[0]
            
            if current_face_encoding is not None and len(known_face_encodings) > 0:
                if SOTA_AVAILABLE and SOTA_METRIC == "cosine":
                    face_distances = [dst.find_cosine_distance(known_encoding, current_face_encoding) for known_encoding in known_face_encodings]
                else:
                    face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    min_distance = face_distances[best_match_index]
                    
                    with data_lock: threshold = server_data["recognition_threshold"]
                    if min_distance < threshold:
                        name = known_face_names[best_match_index]
                        confidence = max(0, min(100, (1.0 - (min_distance / MAX_RECOGNITION_DISTANCE)) * 100))
        except Exception as e:
            pass # Failed to process this face
        
        return (face_location, name, confidence)
    # --- <<< End of helper function >>> ---

    # Wait until the models are loaded before starting
    while INITIALIZING and not APP_SHOULD_QUIT:
        time.sleep(0.1)

    while not APP_SHOULD_QUIT:
        
        frame = None
        with data_lock:
            if latest_raw_frame is None:
                # This will be true if the reader thread failed to start
                if VIDEO_THREAD_STARTED == False and INITIALIZING == False:
                    LOADING_STATUS_MESSAGE = "Error: No video source. Reader thread is not running."
                time.sleep(0.1); continue
            frame = latest_raw_frame.copy()
            
        frame = cv2.flip(frame, 1)
        frame = resize_with_aspect_ratio(frame, max_w=FRAME_WIDTH, max_h=FRAME_HEIGHT)
        if frame is None:
            continue
        
        frame_count += 1
        with data_lock:
            is_recording = server_data["is_recording"]
            current_model = server_data["model"]
            current_yolo_conf = server_data["yolo_conf"]
            current_yolo_imgsz = server_data["yolo_imgsz"]
            current_retinaface_conf = server_data["retinaface_conf"]
            current_alignment_mode = server_data["face_alignment_mode"]
            current_enhancement_mode = server_data["face_enhancement_mode"] 
            
        if is_recording:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if last_action_frame_gray is None:
                last_action_frame_gray = gray
            else:
                frame_delta = cv2.absdiff(last_action_frame_gray, gray); thresh = cv2.threshold(frame_delta, ACTION_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
                if thresh.sum() > 0:
                    last_action_frame_gray = gray;
                    
        # --- YOLO-BASED TRACKING (Optimized) ---
        yolo_results = yolo_model.track(
            frame, device=DEVICE, persist=True, classes=[0], 
            conf=current_yolo_conf, imgsz=current_yolo_imgsz, verbose=False
        )
        body_boxes_with_ids = {} 
        if yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                body_boxes_with_ids[track_id] = (t, r, b, l)

        # --- FACE RECOGNITION (Every Nth Frame) ---
        face_processing_results = []
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0:
            for track_id in person_registry:
                if "face_location" in person_registry[track_id]:
                    person_registry[track_id]["face_location"] = None
            try:
                faces = RetinaFace.detect_faces(frame, threshold=current_retinaface_conf, device=DEVICE) 
            except Exception as e: 
                faces = {}
            
            face_locations = []
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    x1, y1, x2, y2 = face_data['facial_area']
                    face_locations.append((int(y1), int(x2), int(y2), int(x1))) # t,r,b,l

            # --- <<< PARALLEL PROCESSING >>> ---
            if face_locations:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                
                # Create a list of jobs to do
                jobs_to_do = []
                for loc in face_locations:
                    jobs_to_do.append((rgb_frame, loc, current_alignment_mode, current_enhancement_mode))
                
                # Run all jobs in parallel
                with ThreadPoolExecutor(max_workers=max(1, len(face_locations))) as executor:
                    # map() is an efficient way to run all jobs and get results
                    face_processing_results = list(executor.map(_process_single_face, jobs_to_do))
            # --- <<< END PARALLEL >>> ---

            # --- Process results sequentially (this part is fast) ---
            for face_location, name, confidence in face_processing_results:
                body_track_id = get_containing_body_box(face_location, body_boxes_with_ids)
                if body_track_id is None:
                    continue 

                if name != "Unknown":
                    person_registry[body_track_id] = {
                        "name": name, 
                        "confidence": confidence, 
                        "face_location": face_location 
                    }
                    current_time = time.time()
                    if (current_model != MODEL_OFF and 
                        (current_time - last_analysis_time.get(name, 0)) > ANALYSIS_COOLDOWN and 
                        not is_recording): 
                        last_analysis_time[name] = current_time
                        analysis_thread = threading.Thread(target=analyze_frame_with_gemma, args=(frame.copy(), name), daemon=True)
                        analysis_thread.start()
                else: 
                    if person_registry.get(body_track_id, {}).get("name", "Person") == "Person":
                        person_registry[body_track_id] = {
                            "name": "Person",
                            "confidence": 0, 
                            "face_location": face_location 
                        }

        # --- STEP E: Drawing & Data Sync (Every Frame) ---
        live_face_payload = []
        with data_lock:
            analysis_snapshot = analysis_results.copy()

        for track_id, (t, r, b, l) in body_boxes_with_ids.items():
            person_info = person_registry.get(track_id, {"name": "Person", "confidence": 0.0, "face_location": None})
            name = person_info["name"]
            confidence = person_info["confidence"]
            face_location = person_info["face_location"] 

            body_color = COLOR_BODY_KNOWN if name != "Person" else COLOR_BODY_UNKNOWN
            t_body, r_body, b_body, l_body = max(0, t-BOX_PADDING), min(frame.shape[1], r+BOX_PADDING), min(frame.shape[0], b+BOX_PADDING), max(0, l-BOX_PADDING)
            cv2.rectangle(frame, (l_body, t_body), (r_body, b_body), body_color, 2)
            
            label_text = f"{name}"
            if confidence > 0 and name != "Person": 
                label_text += f" ({int(confidence)}%)"
            elif name == "Person":
                label_text = "Unknown Person"
            
            font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.7; font_thickness = 2; text_padding = 5 
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            label_top = max(0, t_body - text_height - baseline - text_padding * 2)
            label_bottom = label_top + text_height + baseline + text_padding * 2
            label_left = l_body; label_right = l_body + text_width + text_padding * 2

            cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), body_color, cv2.FILLED)
            cv2.putText(frame, label_text, (l_body + text_padding, label_bottom - text_padding - baseline), font, font_scale, COLOR_TEXT_FOREGROUND, font_thickness)

            if face_location: 
                ft, fr, fb, fl = face_location
                cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

            if current_model == MODEL_OFF or is_recording:
                analysis_text = "<i>(Paused)</i>"
            else:
                analysis_text = analysis_snapshot.get(name, "")
            
            live_face_payload.append({ "name": name, "confidence": int(confidence), "analysis": analysis_text })
        
        with data_lock:
            global output_frame
            output_frame = frame.copy()
            server_data["live_faces"] = live_face_payload
            
    # --- End of thread loop ---
    cv2.destroyAllWindows()
    VIDEO_THREAD_STARTED = False # Mark as stopped
    LOADING_STATUS_MESSAGE = "Processor thread stopped."

# --- TUI HELPER FUNCTIONS ---

def toggle_action_analysis():
    global server_data, data_lock, action_thread, stop_action_thread, action_frames, last_action_frame_gray
    with data_lock:
        server_data["is_recording"] = not server_data["is_recording"]
        current_model = server_data['model'] 
        if server_data["is_recording"]:
            if current_model == MODEL_OFF:
                server_data["action_result"] = "Model is Off. Please select a model to start."
                server_data["is_recording"] = False
            else:
                stop_action_thread = False; action_frames = []; last_action_frame_gray = None 
                server_data["action_result"] = "Live analysis started...\n"; server_data["keyframe_count"] = 0
                action_thread = threading.Thread(target=action_comprehension_thread, daemon=True); action_thread.start()
        else:
            if action_thread is not None:
                stop_action_thread = True; action_thread = None
            if server_data["action_result"] and "stopped" not in server_data["action_result"]:
                server_data["action_result"] += "\n...Live analysis stopped."

def set_yolo_model(model_key):
    global server_data, data_lock, yolo_model, LOADING_STATUS_MESSAGE, DEVICE
    if model_key not in YOLO_MODELS: return
    with data_lock:
        if model_key == server_data['yolo_model_key']: return 
        model_path = YOLO_MODELS.get(model_key) 
        LOADING_STATUS_MESSAGE = f"Loading YOLO model: {model_path}..."
        try:
            def load_yolo_thread():
                global yolo_model, server_data, LOADING_STATUS_MESSAGE
                new_yolo = YOLO(model_path, verbose=False); new_yolo.to(DEVICE)
                yolo_model = new_yolo
                with data_lock: server_data['yolo_model_key'] = model_key
                LOADING_STATUS_MESSAGE = f"YOLO model {model_key} loaded."; time.sleep(1)
                LOADING_STATUS_MESSAGE = ""
            threading.Thread(target=load_yolo_thread, daemon=True).start()
        except Exception as e:
            LOADING_STATUS_MESSAGE = f"Error loading YOLO: {e}"

def toggle_gfpgan():
    global server_data, data_lock, GFPGAN_AVAILABLE
    if not GFPGAN_AVAILABLE: return
    with data_lock:
        server_data['face_enhancement_mode'] = 'on' if server_data['face_enhancement_mode'] == 'off' else 'off'

def toggle_alignment():
    global server_data, data_lock
    with data_lock:
        server_data['face_alignment_mode'] = 'accurate' if server_data['face_alignment_mode'] == 'fast' else 'fast'

# --- NEW: Main Thread "Mode" Functions ---

def run_update_mode():
    print("Attempting to pull latest version from GitHub...")
    print("----------------------------------------------")
    try:
        process = subprocess.Popen(['git', 'pull'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        print(stdout + "\n" + stderr)
    except Exception as e:
        print(f"An error occurred: {e}")
    print("----------------------------------------------")
    input("--- Press Enter to return to TUI ---")
    return "tui"

def run_cv2_window_mode():
    global VIDEO_THREAD_STARTED
    if not VIDEO_THREAD_STARTED:
        print("Video thread is not running. Cannot open window.")
        time.sleep(2); return "tui"
        
    print("Opening CV2 window... Press 'q' in the window to close.")
    window_name = "Headless Feed (Test)"
    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for frame...", (50, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    while True:
        frame_to_show = None
        with data_lock:
            if output_frame is not None:
                frame_to_show = output_frame.copy()
        
        if frame_to_show is None: frame_to_show = placeholder
        try:
            frame_to_show = resize_with_aspect_ratio(frame_to_show, max_w=1280, max_h=720)
            if frame_to_show is not None: cv2.imshow(window_name, frame_to_show)
            else: cv2.imshow(window_name, placeholder)
        except Exception as e:
             cv2.imshow(window_name, placeholder)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        except cv2.error:
            break

    cv2.destroyAllWindows(); cv2.waitKey(1)
    print("CV2 window closed. Returning to TUI...")
    time.sleep(0.5); return "tui"

# --- <<< MODIFIED: Threaded System Initializer >>> ---
def threaded_system_init():
    global INITIALIZING, SYSTEM_INITIALIZED, VIDEO_THREAD_STARTED, LOADING_STATUS_MESSAGE
    global video_thread, reader_thread, CURRENT_STREAM_SOURCE, latest_raw_frame
    
    INITIALIZING = True
    SYSTEM_INITIALIZED = False
    VIDEO_THREAD_STARTED = False
    
    try:
        # --- MODIFIED: Start video threads *first* to check connection ---
        print_progress_bar(0, 6, "Connecting to video stream...")
        latest_raw_frame = None 

        reader_thread = threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True)
        reader_thread.start()

        # Give the reader 5 seconds to connect
        connect_timeout = 5.0
        start_time = time.time()
        while not VIDEO_THREAD_STARTED and (time.time() - start_time) < connect_timeout:
            if not reader_thread.is_alive(): # Reader thread failed and exited
                break
            time.sleep(0.1)

        if not VIDEO_THREAD_STARTED:
            # If it's still not started, it failed
            LOADING_STATUS_MESSAGE += "\n\nConnection failed. Please check IP and restart."
            INITIALIZING = False
            return # Exit the init thread
        
        # --- If connection is good, start processor and load models ---
        video_thread = threading.Thread(target=video_processing_thread, args=(), daemon=True)
        video_thread.start()

        # --- Now, load resources sequentially and with error logging ---
        _load_imports()
        _load_deepface()
        _load_yolo()
        _load_gfpgan()
        _load_faces()
        
        print_progress_bar(6, 6, "System Running.")
        SYSTEM_INITIALIZED = True
        time.sleep(1.0); LOADING_STATUS_MESSAGE = ""
        
    except Exception as e:
        # This will catch any error from the loading functions
        LOADING_STATUS_MESSAGE = f"FATAL ERROR during init:\n{e}\n{traceback.format_exc()}"
        SYSTEM_INITIALIZED = False
    finally:
        INITIALIZING = False


# --- TUI DRAWING FUNCTION ---

def draw_tui(stdscr):
    global server_data, data_lock, APP_SHOULD_QUIT, TUI_INFO_MESSAGE
    global SYSTEM_INITIALIZED, INITIALIZING, LOADING_STATUS_MESSAGE
    global video_thread, reader_thread, VIDEO_THREAD_STARTED, CURRENT_STREAM_SOURCE, latest_raw_frame
    
    stdscr.nodelay(True); stdscr.clear(); curses.curs_set(0)
    try:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE)
    except:
        pass
    
    while not APP_SHOULD_QUIT:
        try:
            max_y, max_x = stdscr.getmaxyx()
            key = stdscr.getch()
            
            if key == ord('q'):
                APP_SHOULD_QUIT = True; return "quit"
            elif key == ord('u') and not INITIALIZING:
                return "update"
            elif key == ord('i') and not SYSTEM_INITIALIZED and not INITIALIZING:
                LOADING_STATUS_MESSAGE = "Starting initialization thread..."
                init_thread = threading.Thread(target=threaded_system_init, daemon=True)
                init_thread.start()

            if SYSTEM_INITIALIZED and not INITIALIZING:
                if key == ord('s'):
                    toggle_action_analysis()
                elif key == ord('o'):
                    return "open_window"
                elif key == ord('v'):
                    stdscr.addstr(max_y - 1, 0, "Restarting video stream... Please wait.", curses.color_pair(5))
                    stdscr.refresh()
                    
                    APP_SHOULD_QUIT = True
                    if reader_thread is not None and reader_thread.is_alive():
                        reader_thread.join(timeout=1.0)
                    if video_thread is not None and video_thread.is_alive():
                        video_thread.join(timeout=1.0) 
                    
                    APP_SHOULD_QUIT = False
                    if CURRENT_STREAM_SOURCE == STREAM_WEBCAM:
                        CURRENT_STREAM_SOURCE = STREAM_PI_RTSP
                    else:
                        CURRENT_STREAM_SOURCE = STREAM_WEBCAM
                    
                    with data_lock: latest_raw_frame = None 
                    
                    reader_thread = threading.Thread(target=_frame_reader_loop, args=(CURRENT_STREAM_SOURCE,), daemon=True)
                    reader_thread.start()
                    video_thread = threading.Thread(target=video_processing_thread, args=(), daemon=True)
                    video_thread.start()
                    stdscr.clear()
                elif key == ord('1'): set_yolo_model('n')
                elif key == ord('2'): set_yolo_model('s')
                elif key == ord('3'): set_yolo_model('m')
                elif key == ord('4'): set_yolo_model('x')
                elif key == ord('5'): toggle_gfpgan()
                elif key == ord('6'): toggle_alignment()
            
            stdscr.clear()

            if INITIALIZING:
                stdscr.addstr(0, 0, "🤖 Initializing System...", curses.color_pair(3) | curses.A_BOLD)
                y = 2
                # --- MODIFIED: Show multi-line error messages ---
                for line in LOADING_STATUS_MESSAGE.splitlines():
                    if y >= max_y - 2: break
                    stdscr.addstr(y, 0, line[:max_x-1]); y += 1
                
                if "FATAL ERROR" in LOADING_STATUS_MESSAGE:
                    stdscr.addstr(y+1, 0, "Initialization failed. Press 'q' to quit.")
                elif "Connection failed" in LOADING_STATUS_MESSAGE:
                     stdscr.addstr(y+1, 0, "Press 'q' to quit or 'i' to retry.")

            elif not SYSTEM_INITIALIZED:
                stdscr.addstr(0, 0, "🤖 Headless Face Comprehension", curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(1, 0, TUI_INFO_MESSAGE, curses.A_DIM)
                stdscr.addstr(3, 0, "System is IDLE.", curses.A_DIM)
                stdscr.addstr(5, 0, "Press [i] to Initialize System.", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(6, 0, "Press [u] to Self-Update (git pull).", curses.color_pair(2))
                stdscr.addstr(7, 0, "Press [q] to Quit.", curses.color_pair(2))
                if LOADING_STATUS_MESSAGE and ("Error" in LOADING_STATUS_MESSAGE or "failed" in LOADING_STATUS_MESSAGE):
                    stdscr.addstr(9, 0, "LAST STATUS:", curses.color_pair(4) | curses.A_BOLD)
                    y = 10
                    for line in LOADING_STATUS_MESSAGE.splitlines():
                        if y >= max_y - 2: break
                        stdscr.addstr(y, 0, line[:max_x-1], curses.color_pair(4)); y += 1

            else:
                with data_lock: local_data = server_data.copy()
                
                stdscr.addstr(0, 0, "🤖 Headless Face Comprehension", curses.color_pair(1) | curses.A_BOLD)
                device_str = f"({TUI_INFO_MESSAGE})"; controls = "[1-4] YOLO | [5] GFPGAN | [6] Align | [V]ideo Src | [S]top Analysis | [O]pen Window | [U]pdate | [Q]uit"
                stdscr.addstr(1, 0, device_str, curses.A_DIM); stdscr.addstr(2, 0, controls[:max_x-1]); stdscr.addstr(3, 0, "─" * (max_x - 1))

                yolo_map = {'n': 'Nano', 's': 'Small', 'm': 'Medium', 'x': 'Ultra'}
                yolo_str = yolo_map.get(local_data['yolo_model_key'], 'Unknown')
                gfp_str = "ON" if local_data['face_enhancement_mode'] == 'on' else "OFF"
                align_str = "Accurate" if local_data['face_alignment_mode'] == 'accurate' else "Fast"
                source_str = "Pi RTSP" if CURRENT_STREAM_SOURCE == STREAM_PI_RTSP else "Webcam"
                status_source = f" Source: {source_str} "; status_yolo = f" YOLO: {yolo_str} "; status_gfp = f" GFPGAN: {gfp_str} "; status_align = f" Align: {align_str} "
                
                col = 1
                stdscr.addstr(4, col, status_source, curses.A_REVERSE if CURRENT_STREAM_SOURCE == STREAM_PI_RTSP else curses.A_NORMAL); col += len(status_source) + 1
                stdscr.addstr(4, col, status_yolo, curses.A_REVERSE if local_data['yolo_model_key'] != 'm' else curses.A_NORMAL); col += len(status_yolo) + 1
                stdscr.addstr(4, col, status_gfp, curses.A_REVERSE if local_data['face_enhancement_mode'] == 'on' else curses.A_NORMAL); col += len(status_gfp) + 1
                stdscr.addstr(4, col, status_align, curses.A_REVERSE if local_data['face_alignment_mode'] == 'accurate' else curses.A_NORMAL)
                
                panel_width = max_x // 2; panel_start_y = 6
                
                stdscr.addstr(panel_start_y, 0, "👤 Detected People", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(panel_start_y + 1, 0, "─" * (panel_width - 2))
                live_faces = local_data.get('live_faces', [])
                
                if not VIDEO_THREAD_STARTED:
                    stdscr.addstr(panel_start_y + 2, 1, "Video thread is NOT running.", curses.color_pair(4))
                    if LOADING_STATUS_MESSAGE and "Error" in LOADING_STATUS_MESSAGE:
                         stdscr.addstr(panel_start_y + 3, 1, LOADING_STATUS_MESSAGE[:panel_width-2], curses.color_pair(4))
                elif not live_faces:
                    stdscr.addstr(panel_start_y + 2, 1, "No persons detected.", curses.A_DIM)
                else:
                    for i, face in enumerate(live_faces):
                        if (panel_start_y + 2) + i >= max_y - 2: break 
                        name = face.get('name', 'Unknown'); conf = face.get('confidence', 0)
                        if name == "Person":
                            display_name = "Unknown Person"
                            stdscr.addstr(panel_start_y + 2 + i, 1, display_name, curses.color_pair(3))
                        else:
                            display_name = f"{name} ({conf}%)"
                            stdscr.addstr(panel_start_y + 2 + i, 1, display_name, curses.color_pair(2) | curses.A_BOLD)

                stdscr.addstr(panel_start_y, panel_width, "🔳 Action Analysis", curses.color_pair(3) | curses.A_BOLD)
                stdscr.addstr(panel_start_y + 1, panel_width, "─" * (panel_width - 1))
                analysis_state = "RECORDING" if local_data['is_recording'] else "STOPPED"
                analysis_color = curses.color_pair(4) | curses.A_BOLD if local_data['is_recording'] else curses.A_DIM
                stdscr.addstr(panel_start_y + 2, panel_width + 1, f"Status: {analysis_state} (Keyframes: {local_data['keyframe_count']})", analysis_color)
                
                action_result = local_data.get('action_result', "Waiting for analysis...")
                log_lines = action_result.splitlines()
                if not log_lines:
                     stdscr.addstr(panel_start_y + 3, panel_width + 1, "...", curses.A_DIM)
                start_line_idx = max(0, len(log_lines) - (max_y - (panel_start_y + 4))) 
                draw_y = panel_start_y + 3
                for line in log_lines[start_line_idx:]:
                    if draw_y >= max_y - 1: break
                    stdscr.addstr(draw_y, panel_width + 1, line[:panel_width-2], curses.A_DIM); draw_y += 1
                
                if LOADING_STATUS_MESSAGE and not INITIALIZING:
                     stdscr.addstr(max_y - 1, 0, LOADING_STATUS_MESSAGE[:max_x-1], curses.color_pair(5))
            
            stdscr.refresh()
            time.sleep(0.05)
            
        except curses.error:
            stdscr.clear()
        except KeyboardInterrupt:
            APP_SHOULD_QUIT = True; return "quit"
    
    return "quit"

# --- Main entry point ---
if __name__ == "__main__":
    next_action = "tui"
    try:
        while next_action != "quit":
            if next_action == "tui":
                next_action = curses.wrapper(draw_tui)
            elif next_action == "open_window":
                next_action = run_cv2_window_mode()
            elif next_action == "update":
                next_action = run_update_mode()
    finally:
        APP_SHOULD_QUIT = True
        if reader_thread is not None and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
        if video_thread is not None and video_thread.is_alive():
            video_thread.join(timeout=1.0) 
        cv2.destroyAllWindows()
        # --- MODIFIED: Print a clean exit message ---
        # The 'Application shut down cleanly' was part of the problem.
        # Now we print a specific exit message.
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Application shut down.")