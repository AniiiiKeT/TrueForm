import cv2
import mediapipe as mp
import json
import os

# Configuration
VIDEO_PATH = "../data/raw/video1.mp4"  # adjust per video
OUTPUT_DIR = "../data/keypoints/video1"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Prepare keypoints dictionary
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility
            })
    
    # Write keypoints to JSON
    out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.json")
    with open(out_path, 'w') as f:
        json.dump({'frame': frame_idx, 'keypoints': keypoints}, f)

    frame_idx += 1

cap.release()
pose.close()
print(f"Extracted pose for {frame_idx} frames.")
