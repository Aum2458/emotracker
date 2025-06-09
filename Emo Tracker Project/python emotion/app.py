from flask import Flask, render_template, Response, jsonify
import cv2
from deepface import DeepFace

app = Flask(__name__)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Store the latest detected emotion
emotion_result = {"emotion": "neutral"}  

def gen_frames():
    global emotion_result  # Use global to update the latest emotion
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if no frame

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Perform emotion analysis
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                emotion_result = {"emotion": emotion}  # Update the latest detected emotion

                # Draw rectangle around face and label with emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except:
                pass  # Ignore errors if face detection fails

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    """Returns the latest detected emotion as JSON."""
    return jsonify(emotion_result)

if __name__ == '__main__':
    app.run(debug=True)
