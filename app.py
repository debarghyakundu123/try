from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

# Initialize Flask application
app = Flask(__name__)

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables
counter = 0
status = True

# Initialize video capture
cap = cv2.VideoCapture(0)

# Calculate angle between points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Get coordinates of body parts
def detection_body_part(landmarks, body_part_name):
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

# Sit-Up exercise logic
class SitUpExercise:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_abdomen(self):
        left_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        left_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        return calculate_angle(left_shoulder, left_hip, left_knee)

    def perform_sit_up(self, counter, status):
        angle = self.angle_of_the_abdomen()
        if status:
            if angle < 55:  # Sit-up completed
                counter += 1
                status = False
        else:
            if angle > 105:  # Reset position
                status = True
        return counter, status

# Flask route to generate video feed and send it to client
@app.route('/video_feed')
def video_feed():
    def generate():
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            global counter, status
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (800, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = pose.process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    sit_up_exercise = SitUpExercise(landmarks)
                    counter, status = sit_up_exercise.perform_sit_up(counter, status)

                # Render score table
                cv2.putText(frame, f"Counter: {counter}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Status: {status}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2)
                )

                # Convert the frame to JPEG format
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route for the home page
@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)
