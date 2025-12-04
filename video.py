import cv2
import threading
import os
import time
from flask import Flask, Response

# --- CONFIGURATION ---
# Replace with your actual RTSP URL
RTSP_URL = "rtsp://admin:mysecretpassword@100.114.210.58:8554/cam"
PORT = 5005

app = Flask(__name__)

# --- GLOBAL STATE ---
output_frame = None
lock = threading.Lock()

def rtsp_capture_thread():
    """
    Runs in a separate thread to read frames as fast as possible.
    This prevents RTSP lag by always keeping the latest frame ready.
    """
    global output_frame
    
    # Critical for low latency RTSP via OpenCV
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
    
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Simple reconnect logic
            print("Stream lost, reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        # Atomic update of the frame
        with lock:
            output_frame = frame

def generate_frames():
    """Generates the multipart JPEG stream for the browser."""
    while True:
        with lock:
            if output_frame is None:
                continue
            # Encode frame to JPEG
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        
        # Yield the standard multipart/x-mixed-replace format
        yield(b'--frame\r\n' 
              b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/')
def index():
    """Minimal HTML page."""
    return "<html><body style='margin:0; background:black; display:flex; justify-content:center; align-items:center; height:100vh;'><img src='/video_feed' style='max-width:100%; max-height:100%;'></body></html>"

@app.route('/video_feed')
def video_feed():
    """Route that serves the image stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the RTSP reader thread
    t = threading.Thread(target=rtsp_capture_thread, daemon=True)
    t.start()
    
    print(f"Server started at http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)