import cv2
import mediapipe as mp
import json
import os

# Configuration
VIDEO_PATH = "data/raw/Video-1.mp4"
OUTPUT_DIR = "data/keypoints/Video-1"
TARGET_RESOLUTION = (1280, 720)
FRAME_SKIP = 1 

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def preprocess_frame(frame, target_resolution):
    """
    Resize and normalize the frame.
    Args:
        frame: BGR image from OpenCV.
        target_resolution: (width, height) tuple.
    Returns:
        preprocessed BGR frame.
    """
    # Resize frame
    frame_resized = cv2.resize(frame, target_resolution)
    # (Optional) Additional steps: denoising, stabilization, color adjustments
    return frame_resized

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Optionally skip frames to speed up processing
    if frame_idx % FRAME_SKIP != 0:
        frame_idx += 1
        continue

    # Preprocess (resize) frame
    frame_proc = preprocess_frame(frame, TARGET_RESOLUTION)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
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
print(f"Extracted pose for {frame_idx} frames at resolution {TARGET_RESOLUTION}.")
