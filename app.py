from flask import Flask, Response, render_template, request
import torch
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Store user-provided IP Webcam URL
IP_WEBCAM_URL = None

def generate_frames():
    global IP_WEBCAM_URL
    if not IP_WEBCAM_URL:
        return

    cap = cv2.VideoCapture(IP_WEBCAM_URL)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    global IP_WEBCAM_URL
    if request.method == 'POST':
        IP_WEBCAM_URL = request.form['ip_url']
    return render_template('index.html', ip_url=IP_WEBCAM_URL)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
