import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import numpy as np

# Directly import the internal solutions modules to bypass top-level API issues
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize the hand detector directly from the native module
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def transform(self, frame):
        # 1. Convert WebRTC frame to OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Convert to RGB for MediaPipe processing
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)

        label = "No Hand Detected"

        # 3. If a hand is found, calculate the gesture
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the skeletal connections on screen
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the coordinates of specific tracking points
                landmarks = hand_landmarks.landmark
                fingers = []
                
                # Thumb (checks if tip is to the right/left of the knuckle)
                if landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 4 Fingers: Index, Middle, Ring, Pinky
                finger_tips = [
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP
                ]
                finger_pips = [
                    mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp_hands.HandLandmark.RING_FINGER_PIP,
                    mp_hands.HandLandmark.PINKY_PIP
                ]
                
                for tip, pip in zip(finger_tips, finger_pips):
                    if landmarks[tip].y < landmarks[pip].y: # Y goes down in computer vision
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                total_fingers = fingers.count(1)
                
                # Define simple structural rules for gestures
                if total_fingers == 0:
                    label = "Gesture: Fist ✊"
                elif total_fingers == 5:
                    label = "Gesture: Open Palm 🖐️"
                elif total_fingers == 2:
                    label = "Gesture: peace ✌️"
                else:
                    label = f"Fingers Raised: {total_fingers}"

        # 4. Render text overlay onto the video stream
        cv2.putText(img, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return img

# --- Streamlit Layout ---
st.set_page_config(page_title="Hand Gesture App", layout="centered")
st.title("🖐️ Instant Hand Gesture Recognition")
st.write("Show your hand to the camera to see gesture detection in real-time.")

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="gesture-detection",
    video_transformer_factory=GestureTransformer,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)
