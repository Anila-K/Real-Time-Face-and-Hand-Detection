from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Flask app
app = Flask(__name__)

# Initialize streaming flag
streaming = False

# Setup Mediapipe Hands and Face Detection
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

def generate_frames():
    global streaming
    cap = cv2.VideoCapture(0)

    while streaming:
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Hands
        result_hands = hands.process(frame_rgb)

        # Draw hand landmarks on the frame
        if result_hands.multi_hand_landmarks:
            for hand_landmarks in result_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Process the frame with Mediapipe Face Detection
        result_faces = face_detection.process(frame_rgb)

        # Draw face landmarks on the frame
        if result_faces.detections:
            for detection in result_faces.detections:
                mp_drawing.draw_detection(frame, detection)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    global streaming
    streaming = False
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global streaming
    streaming = True  # Start the stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global streaming
    streaming = False  # Stop the stream
    return render_template('index.html')  # Reload the page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == "__main__":
    app.run(debug=True)
