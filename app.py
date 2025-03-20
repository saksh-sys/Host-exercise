import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("Exercise Tracker using AI 💪")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Exercise counter class
class ExerciseCounter:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.count = 0
        self.stage = None
        
    def update(self, angle, up_thresh, down_thresh):
        if angle > up_thresh:
            self.stage = "up"
        if angle < down_thresh and self.stage == 'up':
            self.stage = "down"
            self.count += 1

# Exercise recommendation system
def recommend_exercises(exercise_name):
    recommendations = {
        "bicep curl": ["Hammer Curl", "Concentration Curl", "Reverse Curl"],
        "squat": ["Lunges", "Leg Press", "Deadlift"],
        "pushup": ["Chest Press", "Tricep Dips", "Shoulder Press"]
    }
    return recommendations.get(exercise_name, ["Keep Going!"])

# Initialize counters
bicep_counter = ExerciseCounter("bicep curl")
squat_counter = ExerciseCounter("squat")
pushup_counter = ExerciseCounter("pushup")

# Streamlit Webcam input
run = st.button("Start Tracking")
frame_window = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            
            if landmarks:
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

                # Update counters
                bicep_counter.update(elbow_angle, up_thresh=160, down_thresh=30)
                squat_counter.update(knee_angle, up_thresh=170, down_thresh=90)
                pushup_counter.update(shoulder_angle, up_thresh=160, down_thresh=45)

                # Detect current exercise
                if elbow_angle < 90:
                    current_exercise = "bicep curl"
                elif knee_angle < 90:
                    current_exercise = "squat"
                elif shoulder_angle < 45 and elbow_angle > 90:
                    current_exercise = "pushup"
                else:
                    current_exercise = "unknown"

                # Get recommendations
                recommendations = recommend_exercises(current_exercise)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display results
            frame_window.image(image)
            st.write(f"**Bicep Curls:** {bicep_counter.count}")
            st.write(f"**Squats:** {squat_counter.count}")
            st.write(f"**Pushups:** {pushup_counter.count}")
            st.write(f"**Recommended:** {', '.join(recommendations)}")

    cap.release()
