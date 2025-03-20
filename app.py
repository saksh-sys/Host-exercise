import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image

st.title("Exercise Tracker using AI 💪")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# Exercise Counter Class
class ExerciseCounter:
    def __init__(self):
        self.count = 0
        self.stage = None
        
    def update(self, angle, up_thresh, down_thresh):
        if angle > up_thresh:
            self.stage = "up"
        if angle < down_thresh and self.stage == 'up':
            self.stage = "down"
            self.count += 1

# Initialize exercise counters
bicep_counter = ExerciseCounter()
squat_counter = ExerciseCounter()
pushup_counter = ExerciseCounter()

# Use Streamlit Camera Input
frame = st.camera_input("Show your workout to the camera")

if frame:
    # Convert frame to OpenCV format
    image = Image.open(frame)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # MediaPipe Pose Detection
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get keypoints
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate Angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            # Update Counters
            bicep_counter.update(elbow_angle, up_thresh=160, down_thresh=30)
            squat_counter.update(knee_angle, up_thresh=170, down_thresh=90)
            pushup_counter.update(shoulder_angle, up_thresh=160, down_thresh=45)

            # Draw landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Convert back to RGB for Streamlit display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Workout Tracking", use_column_width=True)

    # Display exercise counts
    st.write(f"**Bicep Curls:** {bicep_counter.count}")
    st.write(f"**Squats:** {squat_counter.count}")
    st.write(f"**Pushups:** {pushup_counter.count}")
