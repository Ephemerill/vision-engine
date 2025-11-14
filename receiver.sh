#!/bin/bash

# This script receives the MJPEG stream on port 5000,
# and serves it over HTTP on port 8080.

PORT="8080"
BOUNDARY="--boundary"

# --- CHECK FOR NCAT ---
if ! command -v ncat &> /dev/null; then
    echo -e "\033[0;31mError: 'ncat' command not found.\033[0m"
    echo "This script now uses 'ncat' as it's more reliable on macOS."
    echo "Please install it by running:"
    echo -e "\033[0;32mbrew install nmap\033[0m"
    exit 1
fi

echo "Starting PERSISTENT web receiver on http://0.0.0.0:$PORT"

# --- NEW: Wrap in a while true loop ---
while true
do
    echo "Waiting for new client connection on port $PORT..." >&2
    
    # This block pipes the HTTP header and the GStreamer output
    # to ncat, which acts as the web server.
    {
        # Print the HTTP header
        printf "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=%s\r\n\r\n" "$BOUNDARY"
        
        # This message goes to stderr (your console)
        echo -e "\n-----------------------------------------------------" >&2
        echo -e "Client connected! GStreamer is now active." >&2
        echo -e "Waiting for MJPEG stream from Pi on UDP port 5000..." >&2
        echo -e "You will see GStreamer logs below when it connects." >&2
        echo -e "-----------------------------------------------------\n" >&2

        # Start the GStreamer pipeline
        gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, encoding-name=JPEG, payload=96" \
            ! rtpjpegdepay \
            ! multipartmux boundary="$BOUNDARY" \
            ! fdsink fd=1
    } | ncat -l "$PORT" # ncat will exit when the client disconnects

    echo "Client disconnected (Broken pipe is normal). GStreamer stopped." >&2
    echo "Looping to wait for next client." >&2
    sleep 1 #
done