import os
import cv2
import json
import numpy as np

# Configuration
VIDEO_PATH = "data/raw/Video-2.mp4"
KEYPOINT_DIR = "data/keypoints/Video-2"
OUTPUT_PATH = "outputs/Video-2/annotated_pose.mp4"
TARGET_RESOLUTION = (1280, 720)

# Pose connections based on MediaPipe's 33-keypoint model
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32)
]

# Color settings
JOINT_COLOR = (0, 255, 0)
LINE_COLOR = (0, 128, 255)
JOINT_RADIUS = 4
LINE_THICKNESS = 2

# Load keypoints from JSON files
def load_keypoints(frame_idx):
    file_path = os.path.join(KEYPOINT_DIR, f"frame_{frame_idx:06d}.json")
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data.get('keypoints', [])

# Initialize video reader and writer
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, TARGET_RESOLUTION)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, TARGET_RESOLUTION)
    keypoints = load_keypoints(frame_idx)

    # Draw keypoints and connections
    points = []
    for kp in keypoints:
        x_px = int(kp['x'] * TARGET_RESOLUTION[0])
        y_px = int(kp['y'] * TARGET_RESOLUTION[1])
        visibility = kp.get('visibility', 0)
        if visibility > 0.5:
            cv2.circle(frame, (x_px, y_px), JOINT_RADIUS, JOINT_COLOR, -1)
            points.append((x_px, y_px))
        else:
            points.append(None)

    # Ensure points list is the right length before drawing connections
    if len(points) >= max(max(c) for c in POSE_CONNECTIONS) + 1:
        for start_idx, end_idx in POSE_CONNECTIONS:
            if points[start_idx] and points[end_idx]:
                cv2.line(frame, points[start_idx], points[end_idx], LINE_COLOR, LINE_THICKNESS)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Annotated video saved to {OUTPUT_PATH}")
