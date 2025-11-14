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

# --- New SOTA Imports (with better error handling) ---
try:
    # We delay the heavy imports, but we can check for the modules
    # This is just a preliminary check; the real import happens in load_resources
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
        LOADING_STATUS_MESSAGE += "\nAll models and faces loaded!"
# ------------------------------------

# --- Configuration ---
MODEL_GEMMA = 'gemma3:4b'
MODEL_LLAVA = 'llava'
MODEL_MOONDREAM = 'moondream'
MODEL_OFF = 'off'
WEBCAM_INDEX = 0
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

# --- TUI STATE ---
SHOW_CV2_WINDOW = False 
APP_SHOULD_QUIT = False
SYSTEM_INITIALIZED = False  # Has the user run the init?
INITIALIZING = False        # Is the init thread running?
VIDEO_THREAD_STARTED = False # Is the video thread running?
LOADING_STATUS_MESSAGE = "" # For the progress bar
video_thread = None         # Placeholder for the video thread
# -------------------

server_data = {
    "is_recording": False,
    "keyframe_count": 0,
    "action_result": "",
    "live_faces": [],
    "model": MODEL_GEMMA, 
    "yolo_model_key": "m", # Default to Medium
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
    global DeepFace, dst # Need access to the imported modules
    
    # This print will be captured by the loading bar
    # print(f"   - Loading from {known_faces_dir} using {SOTA_MODEL}...") 
    
    if SOTA_MODEL != "Dlib":
        if not os.path.exists(known_faces_dir): return

        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if not os.path.isdir(person_dir) or person_name.startswith('.'): continue
            
            image_count = 0
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
                            embedding = representation[0]["embedding"]
                            known_face_encodings.append(embedding)
                            known_face_names.append(person_name)
                            image_count += 1
                        else:
                            pass # print(f"   - WARNING: No face found in {image_path}")

                    except Exception as e:
                        pass # print(f"   - ERROR loading {image_path} with {SOTA_MODEL}: {e}")
            
            # if image_count > 0:
                # print(f"   - Loaded {image_count} SOTA encodings for {person_name}.")

        if not known_face_encodings:
            # print(f"   - WARNING: SOTA model failed. Retrying with Dlib...")
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
            image_count = 0
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encodings_dlib = face_recognition.face_encodings(face_image)
                        for encoding in face_encodings_dlib:
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                        image_count += 1
                    except Exception as e:
                        pass # print(f"   - ERROR loading {image_path} with Dlib: {e}")
            # if image_count > 0:
                # print(f"   - Loaded {image_count} Dlib images for {person_name}.")

def get_containing_body_box(face_box, body_boxes):
    ft, fr, fb, fl = face_box
    face_center_x = (fl + fr) / 2
    face_center_y = (ft + fb) / 2
    for track_id, (bt, br, bb, bl) in body_boxes.items():
        if bl < face_center_x < br and bt < face_center_y < bb:
            return track_id
    return None

# --- ✨ New Resource Loading Function ---
def load_resources():
    """Loads all ML models and known faces. Called by the init thread."""
    global yolo_model, gfpgan_enhancer, MAX_RECOGNITION_DISTANCE, SOTA_MODEL, SOTA_METRIC, DEFAULT_RECOGNITION_THRESHOLD
    global DeepFace, dst, GFPGANer, SOTA_AVAILABLE, GFPGAN_AVAILABLE
    
    TOTAL_LOAD_STEPS = 4
    print_progress_bar(0, TOTAL_LOAD_STEPS, "Initializing...")
    
    # --- Step 1: Perform Heavy Imports ---
    print_progress_bar(1, TOTAL_LOAD_STEPS, "Importing ML libraries...")
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
            
    time.sleep(0.5) # Let user see the import message

    # --- Step 2: Load SOTA Face Model ---
    print_progress_bar(2, TOTAL_LOAD_STEPS, "Loading SOTA Model (ArcFace)...")
    if SOTA_AVAILABLE:
        try:
            DeepFace.build_model(SOTA_MODEL)
            MAX_RECOGNITION_DISTANCE = dst.find_threshold(SOTA_MODEL, SOTA_METRIC) 
        except Exception as e:
            SOTA_MODEL = "Dlib"
            SOTA_METRIC = "euclidean"
            MAX_RECOGNITION_DISTANCE = 0.6
    else:
        SOTA_MODEL = "Dlib"
        SOTA_METRIC = "euclidean"
        MAX_RECOGNITION_DISTANCE = 0.6
    
    DEFAULT_RECOGNITION_THRESHOLD = MAX_RECOGNITION_DISTANCE
    with data_lock:
        server_data["recognition_threshold"] = DEFAULT_RECOGNITION_THRESHOLD

    # --- Step 3: Load YOLO Model ---
    print_progress_bar(3, TOTAL_LOAD_STEPS, "Loading YOLO Model...")
    yolo_model = YOLO(YOLO_MODELS[server_data['yolo_model_key']], verbose=False)

    # --- Step 4: Load GFPGAN Model ---
    print_progress_bar(4, TOTAL_LOAD_STEPS, "Loading GFPGAN Model...")
    if GFPGAN_AVAILABLE:
        try:
            gfpgan_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
        except Exception as e:
            with data_lock:
                server_data["face_enhancement_mode"] = "off_disabled"
    
    # --- Step 5: Load Known Faces (Combined into step 4 for progress bar) ---
    print_progress_bar(4, TOTAL_LOAD_STEPS, "Loading GFPGAN & Known Faces...")
    load_known_faces(KNOWN_FACES_DIR)
    
    # --- Finish ---
    print_progress_bar(TOTAL_LOAD_STEPS, TOTAL_LOAD_STEPS, "All models and faces loaded!")
    time.sleep(0.5)


# --- SOTA-Aware Video Processing Thread ---
def video_processing_thread():
    global data_lock, output_frame, server_data, SHOW_CV2_WINDOW, APP_SHOULD_QUIT
    global analysis_results, yolo_model, person_registry, gfpgan_enhancer
    global DeepFace, dst, SOTA_AVAILABLE, GFPGAN_AVAILABLE # Need these
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    frame_count = 0
    last_analysis_time = {} 
    
    if not cap.isOpened():
        global LOADING_STATUS_MESSAGE
        LOADING_STATUS_MESSAGE = f"Error: Could not open webcam {WEBCAM_INDEX}. Stopping."
        APP_SHOULD_QUIT = True
        return

    while not APP_SHOULD_QUIT:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1.0); continue
            
        frame = cv2.flip(frame, 1)
        frame_count += 1

        with data_lock:
            is_recording = server_data["is_recording"]
            current_model = server_data["model"]
            current_yolo_conf = server_data["yolo_conf"]
            current_yolo_imgsz = server_data["yolo_imgsz"]
            current_retinaface_conf = server_data["retinaface_conf"]
            current_recognition_threshold = server_data["recognition_threshold"] 
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
                    
        # --- YOLO-BASED TRACKING ---
        yolo_results = yolo_model.track(
            frame, 
            persist=True, 
            classes=[0], 
            conf=current_yolo_conf, 
            imgsz=current_yolo_imgsz, 
            verbose=False
        )
        
        body_boxes_with_ids = {} 
        if yolo_results[0].boxes.id is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = yolo_results[0].boxes.id.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                l, t, r, b = box
                body_boxes_with_ids[track_id] = (t, r, b, l)

        # --- FACE RECOGNITION (Every Nth Frame) ---
        if frame_count % FACE_RECOGNITION_NTH_FRAME == 0:
            
            face_locations = []
            
            for track_id in person_registry:
                if "face_location" in person_registry[track_id]:
                    person_registry[track_id]["face_location"] = None

            # --- STEP A: DETECT FACE LOCATIONS (RetinaFace Only) ---
            try:
                faces = RetinaFace.detect_faces(frame, threshold=current_retinaface_conf) 
            except Exception as e: 
                faces = {}
            
            if isinstance(faces, dict):
                for face_key, face_data in faces.items():
                    x1, y1, x2, y2 = face_data['facial_area']
                    face_locations.append((int(y1), int(x2), int(y2), int(x1))) # t,r,b,l

            # --- STEP B: ENCODE & COMPARE FACES ---
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

            for face_location in face_locations:
                name, confidence = "Unknown", 0
                current_face_encoding = None

                # --- Get SOTA Encoding ---
                if SOTA_MODEL != "Dlib":
                    try:
                        t, r, b, l = face_location
                        t_pad = max(0, t - BOX_PADDING)
                        b_pad = min(frame.shape[0], b + BOX_PADDING)
                        l_pad = max(0, l - BOX_PADDING)
                        r_pad = min(frame.shape[1], r + BOX_PADDING)
                        
                        face_crop_rgb = rgb_frame[t_pad:b_pad, l_pad:r_pad]
                        
                        if face_crop_rgb.size == 0: continue
                        
                        input_face = face_crop_rgb 

                        # --- GFPGAN Enhancement Step ---
                        if GFPGAN_AVAILABLE and current_enhancement_mode == "on" and gfpgan_enhancer is not None:
                            try:
                                face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
                                _, _, restored_face_image = gfpgan_enhancer.enhance(
                                    face_crop_bgr, 
                                    has_aligned=False, 
                                    only_center_face=True, 
                                    paste_back=True
                                )
                                
                                if restored_face_image is not None:
                                    input_face = cv2.cvtColor(restored_face_image, cv2.COLOR_BGR2RGB)
                            except Exception as e:
                                pass 
                        # --- End of Enhancement Step ---

                        # --- Conditional Alignment (using 'input_face') ---
                        if SOTA_AVAILABLE:
                            if current_alignment_mode == "accurate":
                                representation = DeepFace.represent(
                                    img_path=input_face, 
                                    model_name=SOTA_MODEL,
                                    enforce_detection=True,
                                    detector_backend='retinaface'
                                )
                            else:
                                representation = DeepFace.represent(
                                    img_path=input_face, 
                                    model_name=SOTA_MODEL,
                                    enforce_detection=False,
                                    detector_backend='skip'
                                )
                            current_face_encoding = representation[0]["embedding"]
                    except Exception as e:
                        pass 
                
                # --- Get Dlib Encoding (Fallback) ---
                else:
                    encodings = face_recognition.face_encodings(rgb_frame, [face_location])
                    if encodings:
                        current_face_encoding = encodings[0]
                
                # --- STEP C: COMPARE ENCODINGS ---
                if current_face_encoding is not None and len(known_face_encodings) > 0:
                    
                    if SOTA_AVAILABLE and SOTA_METRIC == "cosine":
                        face_distances = [dst.find_cosine_distance(known_encoding, current_face_encoding) for known_encoding in known_face_encodings]
                    else:
                        face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        min_distance = face_distances[best_match_index]
                        
                        if min_distance < current_recognition_threshold:
                            name = known_face_names[best_match_index]
                            confidence = max(0, min(100, (1.0 - (min_distance / MAX_RECOGNITION_DISTANCE)) * 100))

                # --- STEP D: ASSOCIATE & ANALYZE ---
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
                
                else: # This is an UNKNOWN face
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
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_padding = 5 
            
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            label_top = max(0, t_body - text_height - baseline - text_padding * 2)
            label_bottom = label_top + text_height + baseline + text_padding * 2
            label_left = l_body
            label_right = l_body + text_width + text_padding * 2

            cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), body_color, cv2.FILLED)
            cv2.putText(frame, label_text, (l_body + text_padding, label_bottom - text_padding - baseline), font, font_scale, COLOR_TEXT_FOREGROUND, font_thickness)

            if face_location: 
                ft, fr, fb, fl = face_location
                cv2.rectangle(frame, (fl, ft), (fr, fb), COLOR_FACE_BOX, 2)

            if current_model == MODEL_OFF or is_recording:
                analysis_text = "<i>(Paused)</i>"
            else:
                analysis_text = analysis_snapshot.get(name, "")
            
            live_face_payload.append({
                "name": name,
                "confidence": int(confidence),
                "analysis": analysis_text
            })
        
        with data_lock:
            global output_frame
            output_frame = frame.copy()
            server_data["live_faces"] = live_face_payload

        # --- TUI CV2 Window ---
        if SHOW_CV2_WINDOW:
            try:
                cv2.imshow("Headless Feed (Test)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    SHOW_CV2_WINDOW = False
                    cv2.destroyWindow("Headless Feed (Test)")
            except cv2.error:
                # Window was closed manually
                SHOW_CV2_WINDOW = False
        else:
            cv2.destroyWindow("Headless Feed (Test)") 
            
    # --- End of thread loop ---
    cap.release()
    cv2.destroyAllWindows()
    # print("Video processing thread stopped.") # Silenced for TUI

# --- TUI HELPER FUNCTIONS (Replaced Flask routes) ---

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
                stop_action_thread = False
                action_frames = [] 
                last_action_frame_gray = None 
                server_data["action_result"] = "Live analysis started...\n"
                server_data["keyframe_count"] = 0
                action_thread = threading.Thread(target=action_comprehension_thread, daemon=True)
                action_thread.start()
        else:
            if action_thread is not None:
                stop_action_thread = True 
                action_thread = None
            if server_data["action_result"] and "stopped" not in server_data["action_result"]:
                server_data["action_result"] += "\n...Live analysis stopped."

def set_yolo_model(model_key):
    global server_data, data_lock, yolo_model, LOADING_STATUS_MESSAGE
    if model_key not in YOLO_MODELS:
        return

    with data_lock:
        if model_key == server_data['yolo_model_key']:
            return 
        
        model_path = YOLO_MODELS.get(model_key) 
        LOADING_STATUS_MESSAGE = f"Loading YOLO model: {model_path}..."
        try:
            # This is the slow part, run in a thread to not block TUI
            def load_yolo_thread():
                global yolo_model, server_data, LOADING_STATUS_MESSAGE
                new_yolo = YOLO(model_path, verbose=False)
                yolo_model = new_yolo
                with data_lock:
                    server_data['yolo_model_key'] = model_key
                LOADING_STATUS_MESSAGE = f"YOLO model {model_key} loaded."
                time.sleep(1)
                LOADING_STATUS_MESSAGE = "" # Clear message
            
            threading.Thread(target=load_yolo_thread, daemon=True).start()

        except Exception as e:
            LOADING_STATUS_MESSAGE = f"Error loading YOLO: {e}"
            pass

def toggle_gfpgan():
    global server_data, data_lock, GFPGAN_AVAILABLE
    if not GFPGAN_AVAILABLE:
        return
    
    with data_lock:
        if server_data['face_enhancement_mode'] == 'on':
            server_data['face_enhancement_mode'] = 'off'
        else:
            server_data['face_enhancement_mode'] = 'on'

def toggle_alignment():
    global server_data, data_lock
    with data_lock:
        if server_data['face_alignment_mode'] == 'fast':
            server_data['face_alignment_mode'] = 'accurate'
        else:
            server_data['face_alignment_mode'] = 'fast'

def perform_update(stdscr):
    """Runs git pull and displays output."""
    stdscr.nodelay(False) # Wait for keypress
    stdscr.clear()
    stdscr.addstr(0, 0, "Attempting to pull latest version from GitHub...")
    stdscr.addstr(2, 0, "----------------------------------------------")
    stdscr.refresh()
    
    try:
        process = subprocess.Popen(['git', 'pull'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        output = stdout + "\n" + stderr
        
        y = 4
        for line in output.splitlines():
            if y < curses.LINES - 2:
                stdscr.addstr(y, 0, line[:curses.COLS-1])
                y += 1
            
    except Exception as e:
        stdscr.addstr(4, 0, f"An error occurred: {e}")
        
    stdscr.addstr(curses.LINES - 1, 0, "--- Press any key to return ---")
    stdscr.refresh()
    stdscr.getch()
    stdscr.nodelay(True) # Go back to non-blocking
    stdscr.clear()

# --- NEW Threaded System Initializer ---
def threaded_system_init():
    global INITIALIZING, SYSTEM_INITIALIZED, VIDEO_THREAD_STARTED, LOADING_STATUS_MESSAGE
    global video_thread
    
    INITIALIZING = True
    SYSTEM_INITIALIZED = False
    VIDEO_THREAD_STARTED = False
    
    try:
        load_resources() # This populates LOADING_STATUS_MESSAGE
        SYSTEM_INITIALIZED = True
        LOADING_STATUS_MESSAGE = "System Initialized. Starting video thread..."
        time.sleep(1.0)
        
        video_thread = threading.Thread(target=video_processing_thread, daemon=True)
        video_thread.start()
        VIDEO_THREAD_STARTED = True
        
        LOADING_STATUS_MESSAGE = "System Running."
        time.sleep(1.0) 
        LOADING_STATUS_MESSAGE = "" # Clear status
        
    except Exception as e:
        LOADING_STATUS_MESSAGE = f"Fatal Error on Init: {e}. System stopped."
        SYSTEM_INITIALIZED = False
    finally:
        INITIALIZING = False # We are no longer in the "initializing" state.


# --- TUI DRAWING FUNCTION ---

def draw_tui(stdscr):
    global server_data, data_lock, SHOW_CV2_WINDOW, APP_SHOULD_QUIT
    global SYSTEM_INITIALIZED, INITIALIZING, LOADING_STATUS_MESSAGE
    
    # --- Curses setup ---
    stdscr.nodelay(True) # Non-blocking getch
    stdscr.clear()
    curses.curs_set(0) # Hide cursor
    
    # Define color pairs
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_WHITE) # Inverted for status
    
    # --- TUI Loop ---
    while not APP_SHOULD_QUIT:
        try:
            # --- Get Window Size ---
            max_y, max_x = stdscr.getmaxyx()
            
            # --- Get Key Press ---
            key = stdscr.getch()
            
            # --- Handle Global Keys (Work anytime) ---
            if key == ord('q'):
                APP_SHOULD_QUIT = True
                break
            elif key == ord('u') and not INITIALIZING:
                perform_update(stdscr)
            elif key == ord('i') and not SYSTEM_INITIALIZED and not INITIALIZING:
                LOADING_STATUS_MESSAGE = "Starting initialization thread..."
                init_thread = threading.Thread(target=threaded_system_init, daemon=True)
                init_thread.start()

            # --- Handle System-Active Keys ---
            if SYSTEM_INITIALIZED and not INITIALIZING:
                if key == ord('s'):
                    toggle_action_analysis()
                elif key == ord('o'):
                    SHOW_CV2_WINDOW = not SHOW_CV2_WINDOW
                    if not SHOW_CV2_WINDOW:
                        cv2.destroyAllWindows()
                elif key == ord('1'):
                    set_yolo_model('n')
                elif key == ord('2'):
                    set_yolo_model('s')
                elif key == ord('3'):
                    set_yolo_model('m')
                elif key == ord('4'):
                    set_yolo_model('x')
                elif key == ord('5'):
                    toggle_gfpgan()
                elif key == ord('6'):
                    toggle_alignment()
            
            # --- TUI DRAWING LOGIC ---
            stdscr.clear()

            # --- 1. Draw "Initializing" Screen ---
            if INITIALIZING:
                stdscr.addstr(0, 0, "🤖 Initializing System...", curses.color_pair(3) | curses.A_BOLD)
                # Draw the multi-line progress bar
                y = 2
                for line in LOADING_STATUS_MESSAGE.splitlines():
                    if y >= max_y - 2: break
                    stdscr.addstr(y, 0, line[:max_x-1])
                    y += 1
                stdscr.addstr(max_y - 1, 0, "Please wait... (This can take a minute)")

            # --- 2. Draw "Idle" Screen ---
            elif not SYSTEM_INITIALIZED:
                stdscr.addstr(0, 0, "🤖 Headless Face Comprehension", curses.color_pair(1) | curses.A_BOLD)
                stdscr.addstr(2, 0, "System is IDLE.", curses.A_DIM)
                stdscr.addstr(4, 0, "Press [i] to Initialize System.", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(5, 0, "Press [u] to Self-Update (git pull).", curses.color_pair(2))
                stdscr.addstr(6, 0, "Press [q] to Quit.", curses.color_pair(2))
                
                # Show error message if init failed
                if LOADING_STATUS_MESSAGE and "Error" in LOADING_STATUS_MESSAGE:
                    stdscr.addstr(8, 0, "LAST ERROR:", curses.color_pair(4) | curses.A_BOLD)
                    stdscr.addstr(9, 0, LOADING_STATUS_MESSAGE[:max_x-1], curses.color_pair(4))

            # --- 3. Draw "Running" Screen ---
            else:
                # --- Get Data Snapshot ---
                with data_lock:
                    local_data = server_data.copy()
                
                # --- 3a. Header & Controls ---
                stdscr.addstr(0, 0, "🤖 Headless Face Comprehension", curses.color_pair(1) | curses.A_BOLD)
                controls = "[1-4] YOLO | [5] GFPGAN | [6] Align | [S]tart/Stop Analysis | [O]pen Window | [U]pdate | [Q]uit"
                stdscr.addstr(1, 0, controls[:max_x-1])
                stdscr.addstr(2, 0, "─" * (max_x - 1))

                # --- 3b. Status Panel ---
                yolo_map = {'n': 'Nano', 's': 'Small', 'm': 'Medium', 'x': 'Ultra'}
                yolo_str = yolo_map.get(local_data['yolo_model_key'], 'Unknown')
                gfp_str = "ON" if local_data['face_enhancement_mode'] == 'on' else "OFF"
                align_str = "Accurate" if local_data['face_alignment_mode'] == 'accurate' else "Fast"
                win_str = "ON" if SHOW_CV2_WINDOW else "OFF"
                
                status_yolo = f" YOLO: {yolo_str} "
                status_gfp = f" GFPGAN: {gfp_str} "
                status_align = f" Align: {align_str} "
                status_win = f" Window: {win_str} "
                
                stdscr.addstr(3, 1, status_yolo, curses.A_REVERSE if local_data['yolo_model_key'] != 'm' else curses.A_NORMAL)
                stdscr.addstr(3, len(status_yolo) + 2, status_gfp, curses.A_REVERSE if local_data['face_enhancement_mode'] == 'on' else curses.A_NORMAL)
                stdscr.addstr(3, len(status_yolo) + len(status_gfp) + 4, status_align, curses.A_REVERSE if local_data['face_alignment_mode'] == 'accurate' else curses.A_NORMAL)
                stdscr.addstr(3, len(status_yolo) + len(status_gfp) + len(status_align) + 6, status_win, curses.A_REVERSE if SHOW_CV2_WINDOW else curses.A_NORMAL)
                
                # --- 3c. Main Content Panels ---
                panel_width = max_x // 2
                
                # --- Detected People Panel ---
                stdscr.addstr(5, 0, "👤 Detected People", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(6, 0, "─" * (panel_width - 2))
                
                live_faces = local_data.get('live_faces', [])
                if not live_faces:
                    stdscr.addstr(7, 1, "No persons detected.", curses.A_DIM)
                else:
                    for i, face in enumerate(live_faces):
                        if 7 + i >= max_y - 2: break 
                        name = face.get('name', 'Unknown')
                        conf = face.get('confidence', 0)
                        
                        if name == "Person":
                            display_name = "Unknown Person"
                            stdscr.addstr(7 + i, 1, display_name, curses.color_pair(3))
                        else:
                            display_name = f"{name} ({conf}%)"
                            stdscr.addstr(7 + i, 1, display_name, curses.color_pair(2) | curses.A_BOLD)

                # --- Action Analysis Panel ---
                stdscr.addstr(5, panel_width, "🔳 Action Analysis", curses.color_pair(3) | curses.A_BOLD)
                stdscr.addstr(6, panel_width, "─" * (panel_width - 1))
                
                analysis_state = "RECORDING" if local_data['is_recording'] else "STOPPED"
                analysis_color = curses.color_pair(4) | curses.A_BOLD if local_data['is_recording'] else curses.A_DIM
                stdscr.addstr(7, panel_width + 1, f"Status: {analysis_state} (Keyframes: {local_data['keyframe_count']})", analysis_color)
                
                action_result = local_data.get('action_result', "Waiting for analysis...")
                log_lines = action_result.splitlines()
                if not log_lines:
                     stdscr.addstr(8, panel_width + 1, "...", curses.A_DIM)

                start_line_idx = max(0, len(log_lines) - (max_y - 10)) 
                draw_y = 8
                for line in log_lines[start_line_idx:]:
                    if draw_y >= max_y - 1: break
                    stdscr.addstr(draw_y, panel_width + 1, line[:panel_width-2], curses.A_DIM)
                    draw_y += 1
                
                # --- 3d. Draw Status Message (like YOLO loading) ---
                if LOADING_STATUS_MESSAGE and not INITIALIZING:
                     stdscr.addstr(max_y - 1, 0, LOADING_STATUS_MESSAGE[:max_x-1], curses.color_pair(5))


            # --- Refresh Screen ---
            stdscr.refresh()
            
            # --- Sleep ---
            time.sleep(0.05) # ~20 FPS TUI refresh rate
            
        except curses.error:
            # Handle terminal resize
            stdscr.clear()
        except KeyboardInterrupt:
            APP_SHOULD_QUIT = True

# --- Main entry point ---
if __name__ == "__main__":
    try:
        curses.wrapper(draw_tui)
    finally:
        # Cleanup
        APP_SHOULD_QUIT = True
        if video_thread is not None and video_thread.is_alive():
            video_thread.join(timeout=1.0) # Wait for thread to finish
        cv2.destroyAllWindows()
        print("Application shut down cleanly.")