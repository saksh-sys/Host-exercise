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
            
    def get_count(self):
        return self.count

# Create session state variables for counters
if 'bicep_counter' not in st.session_state:
    st.session_state.bicep_counter = ExerciseCounter()
if 'squat_counter' not in st.session_state:
    st.session_state.squat_counter = ExerciseCounter()
if 'pushup_counter' not in st.session_state:
    st.session_state.pushup_counter = ExerciseCounter()

# Video Processing Class for Streamlit WebRTC
class PoseProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Get frame dimensions
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
            st.session_state.bicep_counter.update(elbow_angle, up_thresh=160, down_thresh=30)
            st.session_state.squat_counter.update(knee_angle, up_thresh=170, down_thresh=90)
            st.session_state.pushup_counter.update(shoulder_angle, up_thresh=160, down_thresh=45)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display exercise counts on screen with a background to make text more readable
        # Create a semi-transparent overlay for the counts
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Add text with counts
        cv2.putText(img, f"Bicep Curls: {st.session_state.bicep_counter.get_count()}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Squats: {st.session_state.squat_counter.get_count()}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Pushups: {st.session_state.pushup_counter.get_count()}", 
                   (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# Create a proper RTC configuration
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Instructions for users
st.markdown("""
### How to use:
1. Click the 'START' button below to activate your webcam
2. Position yourself so your whole body is visible
3. Start exercising - the app will count your reps automatically!
""")

# Start real-time video processing with proper configuration
webrtc_ctx = webrtc_streamer(
    key="exercise-tracker",
    video_processor_factory=PoseProcessor,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)

# Reset button for counters
if st.button("Reset Counters"):
    st.session_state.bicep_counter = ExerciseCounter()
    st.session_state.squat_counter = ExerciseCounter()
    st.session_state.pushup_counter = ExerciseCounter()
    st.experimental_rerun()
