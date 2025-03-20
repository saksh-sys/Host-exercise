import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import time

# Page configuration
st.set_page_config(
    page_title="AI Fitness Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # End point
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is within domain of arccos
    
    # Calculate angle in degrees
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return angle

# Exercise tracker class
class ExerciseCounter:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.count = 0
        self.stage = None
        self.prev_stage = None
        self.confidence = 0  # Confidence in current exercise detection
        self.consecutive_detections = 0  # For stability
        self.last_count_time = time.time()  # To prevent counting too quickly
        
    def update(self, angle, up_thresh, down_thresh, confidence_boost=0):
        current_time = time.time()
        time_since_last_count = current_time - self.last_count_time
        
        # Update exercise detection confidence
        self.confidence = min(1.0, self.confidence + confidence_boost)
        
        # Update stage based on angle
        prev_stage = self.stage
        
        if angle > up_thresh:
            self.stage = "up"
        elif angle < down_thresh:
            self.stage = "down"
            
        # Only count if we've moved from up to down with sufficient confidence
        # and enough time has passed (to prevent double counting)
        if self.stage == "down" and prev_stage == "up" and self.confidence > 0.7 and time_since_last_count > 1.0:
            self.count += 1
            self.last_count_time = current_time
            
        # If stage hasn't changed, reduce confidence slightly
        if self.stage == prev_stage:
            self.confidence = max(0, self.confidence - 0.01)
            
    def reset(self):
        self.count = 0
        self.stage = None
        self.prev_stage = None
        self.confidence = 0
        self.consecutive_detections = 0

# Exercise recommendation system
def recommend_exercises(exercise_name):
    recommendations = {
        "bicep curl": ["Hammer Curls", "Concentration Curls", "Reverse Curls", "Try for 3 sets of 12 reps"],
        "squat": ["Lunges", "Leg Press", "Deadlifts", "Try for 3 sets of 15 reps"],
        "pushup": ["Chest Press", "Tricep Dips", "Shoulder Press", "Try for 3 sets of 10 reps"],
        "shoulder press": ["Lateral Raises", "Front Raises", "Reverse Flys", "Try for 3 sets of 12 reps"]
    }
    return recommendations.get(exercise_name, ["Get into position to start tracking!"])

# Function to determine current exercise based on body pose
def identify_exercise(landmarks, image_shape):
    # Extract key landmarks
    try:
        # Left side landmarks
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, 
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, 
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        # Right side landmarks
        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        
        # Calculate joint angles for both sides
        l_elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
        
        l_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
        
        l_shoulder_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_shoulder_angle = calculate_angle(r_hip, r_shoulder, r_elbow)
        
        # Use the average of left and right angles for better accuracy
        elbow_angle = (l_elbow_angle + r_elbow_angle) / 2
        knee_angle = (l_knee_angle + r_knee_angle) / 2
        hip_angle = (l_hip_angle + r_hip_angle) / 2
        shoulder_angle = (l_shoulder_angle + r_shoulder_angle) / 2
        
        # Check visibility of key landmarks
        # Note: Visibility might not be available in older versions of mediapipe
        visibility_threshold = 0.65
        
        # Handle potential compatibility issues with different MediaPipe versions
        try:
            bicep_visibility = min(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility or 0
            )
            
            squat_visibility = min(
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility or 0
            )
            
            pushup_visibility = min(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility or 0,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility or 0
            )
        except AttributeError:
            # If visibility is not available, assume high visibility
            bicep_visibility = 1.0
            squat_visibility = 1.0
            pushup_visibility = 1.0
        
        # Feature combinations for exercise identification
        exercises = {
            "bicep curl": {
                "conditions": [
                    elbow_angle < 90,
                    bicep_visibility > visibility_threshold,
                    abs(l_shoulder[1] - r_shoulder[1]) < 0.1,  # Shoulders roughly level
                    abs(l_hip[1] - l_shoulder[1]) > 0.2  # Standing upright
                ],
                "confidence": sum([
                    0.5 if elbow_angle < 90 else 0,
                    0.3 if bicep_visibility > visibility_threshold else 0,
                    0.1 if abs(l_shoulder[1] - r_shoulder[1]) < 0.1 else 0,
                    0.1 if abs(l_hip[1] - l_shoulder[1]) > 0.2 else 0
                ])
            },
            "squat": {
                "conditions": [
                    knee_angle < 140,
                    knee_angle > 70,  # Not too deep to avoid detecting when sitting
                    squat_visibility > visibility_threshold,
                    hip_angle < 120,  # Hip bend
                    abs(l_knee[0] - r_knee[0]) < 0.15  # Knees roughly aligned
                ],
                "confidence": sum([
                    0.4 if knee_angle < 140 and knee_angle > 70 else 0,
                    0.3 if squat_visibility > visibility_threshold else 0,
                    0.2 if hip_angle < 120 else 0,
                    0.1 if abs(l_knee[0] - r_knee[0]) < 0.15 else 0
                ])
            },
            "pushup": {
                "conditions": [
                    shoulder_angle < 80,
                    elbow_angle < 120,
                    pushup_visibility > visibility_threshold,
                    abs(l_shoulder[1] - l_hip[1]) < 0.2  # Body roughly horizontal
                ],
                "confidence": sum([
                    0.4 if shoulder_angle < 80 else 0,
                    0.3 if elbow_angle < 120 else 0,
                    0.2 if pushup_visibility > visibility_threshold else 0,
                    0.1 if abs(l_shoulder[1] - l_hip[1]) < 0.2 else 0
                ])
            }
        }
        
        # Find exercise with highest confidence
        max_confidence = 0
        detected_exercise = "unknown"
        for exercise, data in exercises.items():
            if data["confidence"] > max_confidence and all(data["conditions"]):
                max_confidence = data["confidence"]
                detected_exercise = exercise
                
        # Return detected exercise and joint angles
        return detected_exercise, {
            "elbow_angle": elbow_angle,
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "shoulder_angle": shoulder_angle
        }, max_confidence
        
    except Exception as e:
        return "unknown", {}, 0

# Video processor class for streamlit-webrtc
class PoseProcessor(VideoProcessorBase):
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        # Use model_complexity=1 if available, otherwise use default parameters
        try:
            self.pose = mp_pose.Pose(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                model_complexity=1
            )
        except TypeError:
            # Fallback for older versions
            self.pose = mp_pose.Pose(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
        self.current_exercise = "unknown"
        self.exercise_confidence = 0
        self.angles = {}
        self.frame_count = 0
        
    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Skip some frames for better performance if needed
        if self.frame_count % 2 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        # Resize for better performance
        image = cv2.resize(img, (640, 480))
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Convert back to BGR
        image_rgb.flags.writeable = True
        annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # If pose detected, process it
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Get landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Identify exercise
            self.current_exercise, self.angles, self.exercise_confidence = identify_exercise(landmarks, image.shape)
            
            # Display exercise and confidence
            cv2.putText(
                annotated_image,
                f"Exercise: {self.current_exercise.title()}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            cv2.putText(
                annotated_image,
                f"Confidence: {self.exercise_confidence:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Display angles
            y_pos = 90
            for angle_name, angle_value in self.angles.items():
                cv2.putText(
                    annotated_image,
                    f"{angle_name}: {int(angle_value)}°",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                y_pos += 25
                
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

# Custom CSS for better styling
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-top: 1rem;
    }
    .exercise-counter {
        font-size: 1.2rem;
        padding: 0.5rem;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .recommendations {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .stats-box {
        padding: 1rem;
        background-color: #e3f2fd;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .webcam-container {
        border: 2px solid #2196F3;
        border-radius: 10px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Load CSS
    load_css()
    
    # Main header
    st.markdown('<h1 class="main-header">AI Fitness Tracker 💪</h1>', unsafe_allow_html=True)
    
    # App description
    st.markdown("""
    This app uses computer vision to track your exercises in real-time. 
    It can count repetitions and provide recommendations for your workout routine.
    """)
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">Settings</h2>', unsafe_allow_html=True)
    
    # Model settings
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Exercise settings
    st.sidebar.markdown('<h2 class="sub-header">Exercise Counters</h2>', unsafe_allow_html=True)
    
    # Initialize session state for counters if not exists
    if 'bicep_counter' not in st.session_state:
        st.session_state.bicep_counter = ExerciseCounter("bicep curl")
    if 'squat_counter' not in st.session_state:
        st.session_state.squat_counter = ExerciseCounter("squat")
    if 'pushup_counter' not in st.session_state:
        st.session_state.pushup_counter = ExerciseCounter("pushup")
    
    # Reset button
    if st.sidebar.button('Reset All Counters'):
        st.session_state.bicep_counter.reset()
        st.session_state.squat_counter.reset()
        st.session_state.pushup_counter.reset()
        st.sidebar.success("All counters reset!")
    
    # Individual reset buttons
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button('Reset Biceps'):
        st.session_state.bicep_counter.reset()
    if col2.button('Reset Squats'):
        st.session_state.squat_counter.reset()
    if col3.button('Reset Pushups'):
        st.session_state.pushup_counter.reset()
    
    # Help section
    with st.sidebar.expander("How to use"):
        st.markdown("""
        1. Allow camera access when prompted
        2. Position yourself so your full body is visible
        3. Perform exercises:
           - Bicep curls: Stand straight, curl arms up and down
           - Squats: Stand with feet shoulder-width apart, lower body until thighs are parallel to floor
           - Pushups: Start in plank position, lower body until elbows are at 90 degrees
        4. The app will automatically detect and count your repetitions
        """)
    
    # Credits
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ using Streamlit and MediaPipe")
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    # Webcam feed with streamlit-webrtc
    with col1:
        st.markdown('<h2 class="sub-header">Webcam Feed</h2>', unsafe_allow_html=True)
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Create the WebRTC streamer
        ctx = webrtc_streamer(
            key="pose-detection",
            mode="sendrecv",
            rtc_configuration=rtc_configuration,
            video_processor_factory=lambda: PoseProcessor(
                detection_confidence=detection_confidence,
                tracking_confidence=tracking_confidence
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # If the WebRTC streamer is active, update the exercise counters
        if ctx.video_processor:
            current_exercise = ctx.video_processor.current_exercise
            confidence = ctx.video_processor.exercise_confidence
            angles = ctx.video_processor.angles
            
            # Update relevant counter based on detected exercise
            if current_exercise == "bicep curl" and confidence > 0.7:
                st.session_state.bicep_counter.update(
                    angles.get("elbow_angle", 180), 
                    up_thresh=160, 
                    down_thresh=60,
                    confidence_boost=0.1
                )
            elif current_exercise == "squat" and confidence > 0.7:
                st.session_state.squat_counter.update(
                    angles.get("knee_angle", 180), 
                    up_thresh=160, 
                    down_thresh=90,
                    confidence_boost=0.1
                )
            elif current_exercise == "pushup" and confidence > 0.7:
                st.session_state.pushup_counter.update(
                    angles.get("shoulder_angle", 180), 
                    up_thresh=160, 
                    down_thresh=60,
                    confidence_boost=0.1
                )
    
    # Exercise counters and recommendations
    with col2:
        st.markdown('<h2 class="sub-header">Exercise Counts</h2>', unsafe_allow_html=True)
        
        # Exercise counters
        st.markdown(f"""
        <div class="exercise-counter">
            <strong>Bicep Curls:</strong> {st.session_state.bicep_counter.count}
        </div>
        <div class="exercise-counter">
            <strong>Squats:</strong> {st.session_state.squat_counter.count}
        </div>
        <div class="exercise-counter">
            <strong>Pushups:</strong> {st.session_state.pushup_counter.count}
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<h2 class="sub-header">Workout Recommendations</h2>', unsafe_allow_html=True)
        
        # Only show recommendations if WebRTC is active
        if ctx.video_processor:
            current_exercise = ctx.video_processor.current_exercise
            recommendations = recommend_exercises(current_exercise)
            
            if current_exercise != "unknown":
                st.markdown(f"""
                <div class="recommendations">
                    <strong>Current Exercise:</strong> {current_exercise.title()}<br>
                    <strong>Try Next:</strong> {', '.join(recommendations[:-1])}<br>
                    <strong>Tip:</strong> {recommendations[-1]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendations">
                    Get into position to start an exercise!
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendations">
                Start the webcam to get exercise recommendations.
            </div>
            """, unsafe_allow_html=True)
        
        # Workout stats
        st.markdown('<h2 class="sub-header">Workout Stats</h2>', unsafe_allow_html=True)
        
        # Calculate total reps
        total_reps = (
            st.session_state.bicep_counter.count +
            st.session_state.squat_counter.count +
            st.session_state.pushup_counter.count
        )
        
        # Calculate estimated calories (very rough estimation)
        est_calories = (
            st.session_state.bicep_counter.count * 0.5 +
            st.session_state.squat_counter.count * 0.8 +
            st.session_state.pushup_counter.count * 0.6
        )
        
        st.markdown(f"""
        <div class="stats-box">
            <strong>Total Repetitions:</strong> {total_reps}<br>
            <strong>Estimated Calories:</strong> {est_calories:.1f} kcal<br>
            <strong>Workout Focus:</strong> {
                "Upper Body" if st.session_state.bicep_counter.count + st.session_state.pushup_counter.count > st.session_state.squat_counter.count else
                "Lower Body" if st.session_state.squat_counter.count > st.session_state.bicep_counter.count + st.session_state.pushup_counter.count else
                "Full Body"
            }
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
