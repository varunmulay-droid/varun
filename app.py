from flask import Flask, Response, request, render_template_string
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load YOLOv8 model



# Check if the model exists; otherwise, download it
model_path = "yolov8x.pt"
if not os.path.exists(model_path):
    os.system(f"wget https://github.com/ultralytics/assets/releases/download/v8.0.0/{model_path}")

model = YOLO(model_path)

# HTML template for inputting IP
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Live YOLOv8 Detection</title>
</head>
<body>
    <h2>Enter IP Webcam URL</h2>
    <form action="/video_feed" method="get">
        <label for="ip">IP Webcam URL:</label>
        <input type="text" id="ip" name="ip" required>
        <button type="submit">Start Stream</button>
    </form>
</body>
</html>
"""

def generate_frames(ip_url):
    cap = cv2.VideoCapture(ip_url)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLOv8 object detection
        results = model.predict(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                conf = box.conf[0].item()

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    ip = request.args.get('ip')
    if not ip:
        return "IP Webcam URL is required!", 400
    return Response(generate_frames(ip), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
