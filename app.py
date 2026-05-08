from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model 

app = Flask(__name__)

# ---- Load your trained .h5 model ----
model = load_model("sign_language_model.h5")

labels = ["A", "B", "C", "Hello", "Yes", "No"]
camera = cv2.VideoCapture(0)

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 64, 64, 1)
    return img

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    success, frame = camera.read()
    if not success:
        return jsonify({"prediction": "No Frame"})

    processed = preprocess_frame(frame)
    preds = model.predict(processed)
    predicted_label = labels[np.argmax(preds)]

    return jsonify({"prediction": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
