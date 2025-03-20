import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Exercise tracker
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
            
    def reset(self):
        self.count = 0
        self.stage = None

# Exercise recommendation system
def recommend_exercises(exercise_name):
    recommendations = {
        "bicep curl": ["Hammer Curl", "Concentration Curl", "Reverse Curl"],
        "squat": ["Lunges", "Leg Press", "Deadlift"],
        "pushup": ["Chest Press", "Tricep Dips", "Shoulder Press"]
    }
    return recommendations.get(exercise_name, ["Keep Going!"])

def main():
    # Streamlit UI setup
    st.title('AI Exercise Tracker')
    
    # Sidebar
    st.sidebar.header('Controls')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    
    # Reset button
    if st.sidebar.button('Reset Counters'):
        if 'bicep_counter' in st.session_state:
            st.session_state.bicep_counter.reset()
        if 'squat_counter' in st.session_state:
            st.session_state.squat_counter.reset()
        if 'pushup_counter' in st.session_state:
            st.session_state.pushup_counter.reset()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    # Camera feed placeholder
    with col1:
        stframe = st.empty()
    
    # Counters display
    with col2:
        st.subheader('Exercise Counts')
        bicep_count_text = st.empty()
        squat_count_text = st.empty()
        pushup_count_text = st.empty()
        
        st.subheader('Recommendations')
        recommendations_text = st.empty()
    
    # Initialize session state for counters if not exists
    if 'bicep_counter' not in st.session_state:
        st.session_state.bicep_counter = ExerciseCounter("bicep curl")
    if 'squat_counter' not in st.session_state:
        st.session_state.squat_counter = ExerciseCounter("squat")
    if 'pushup_counter' not in st.session_state:
        st.session_state.pushup_counter = ExerciseCounter("pushup")
    
    # Camera capture
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as pose:
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            current_exercise = "unknown"
            
            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for key landmarks
                try:
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

                    # Calculate joint angles
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    knee_angle = calculate_angle(hip, knee, ankle)
                    hip_angle = calculate_angle(shoulder, hip, knee)
                    shoulder_angle = calculate_angle(hip, shoulder, elbow)
                    
                    # Bicep Curl Counter
                    st.session_state.bicep_counter.update(elbow_angle, up_thresh=160, down_thresh=30)
                    # Squat Counter
                    st.session_state.squat_counter.update(knee_angle, up_thresh=170, down_thresh=90)
                    # Pushup Counter
                    st.session_state.pushup_counter.update(shoulder_angle, up_thresh=160, down_thresh=45)
                    
                    # Determine current exercise based on angles
                    if elbow_angle < 90:
                        current_exercise = "bicep curl"
                    elif knee_angle < 120:
                        current_exercise = "squat"
                    elif shoulder_angle < 45 and elbow_angle > 90:
                        current_exercise = "pushup"
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    # Display angles on image
                    cv2.putText(image, f"Elbow: {int(elbow_angle)}", 
                                (int(elbow[0] * image.shape[1]), int(elbow[1] * image.shape[0])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Knee: {int(knee_angle)}", 
                                (int(knee[0] * image.shape[1]), int(knee[1] * image.shape[0])), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as e:
                    pass  # If landmarks are not detected correctly
            
            # Update counters in UI
            bicep_count_text.markdown(f"**Bicep Curls:** {st.session_state.bicep_counter.count}")
            squat_count_text.markdown(f"**Squats:** {st.session_state.squat_counter.count}")
            pushup_count_text.markdown(f"**Pushups:** {st.session_state.pushup_counter.count}")
            
            # Update recommendations
            recommendations = recommend_exercises(current_exercise)
            if current_exercise != "unknown":
                recommendations_text.markdown(f"**Current Exercise:** {current_exercise.title()}\n\n**Try Next:** {', '.join(recommendations)}")
            else:
                recommendations_text.markdown("Get into position to start an exercise!")
            
            # Display image
            stframe.image(image, channels="BGR", use_column_width=True)
            
            # Check if the Streamlit app is still running
            if not cap.isOpened():
                break
                
            # Small sleep to reduce CPU usage
            time.sleep(0.01)
    
    cap.release()

if __name__ == "__main__":
    main()
