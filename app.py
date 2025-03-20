import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.title("Real-Time AI Exercise Tracker 🏋️‍♂️")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate joint angles
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

# Initialize counters (as global variables)
bicep_counter = ExerciseCounter()
squat_counter = ExerciseCounter()
pushup_counter = ExerciseCounter()

# Video Processing Class for Streamlit WebRTC
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Get image dimensions
        h, w = img.shape[:2]

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract key points
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

            # Calculate angles
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            # Update exercise counters
            bicep_counter.update(elbow_angle, up_thresh=160, down_thresh=30)
            squat_counter.update(knee_angle, up_thresh=170, down_thresh=90)
            pushup_counter.update(shoulder_angle, up_thresh=160, down_thresh=45)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Create a semi-transparent background for the counter text
        overlay = img.copy()
        cv2.rectangle(overlay, (5, 10), (300, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Display exercise counts with improved visibility
        cv2.putText(img, f"Bicep Curls: {bicep_counter.count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Squats: {squat_counter.count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Pushups: {pushup_counter.count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# Configure RTC with explicit options
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Add instructions for users
st.markdown("""
### Instructions:
1. Click the 'START' button below to activate your camera
2. Stand back so your full body is visible
3. Perform exercises and see the counter track your reps
""")

# Start real-time video processing with improved configuration
webrtc_ctx = webrtc_streamer(
    key="exercise-tracker",
    video_processor_factory=PoseProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Add a note about privacy
st.markdown("""
---
**Note:** All processing happens locally in your browser. No video data is sent to any server.
""")
