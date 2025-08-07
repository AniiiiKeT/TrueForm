import os
import json
import cv2
import torch

# Configuration
VIDEO_PATH = "../data/raw/video1.mp4"
OUTPUT_DIR = "../data/equipment/video1"
MODEL_PATH = "../models/yolov5_bow.pt"  # Path to fine-tuned YOLOv5 weights
FRAME_SKIP = 1  # Skip frames to speed up
TARGET_RESOLUTION = (1280, 720)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.conf = 0.25  # confidence threshold
model.iou = 0.45   # NMS IoU threshold


def preprocess_frame(frame, target_resolution):
    """
    Resize and normalize the frame.
    """
    return cv2.resize(frame, target_resolution)


def detect_equipment():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # Preprocess
        frame_resized = preprocess_frame(frame, TARGET_RESOLUTION)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Run detection
        results = model(img_rgb)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        # Prepare list for bow grip and arrow tip
        equipment = []
        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            if label in ['bow_grip', 'arrow_tip']:
                x1, y1, x2, y2 = box
                equipment.append({
                    'label': label,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

        # Write to JSON
        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.json")
        with open(out_path, 'w') as f:
            json.dump({'frame': frame_idx, 'equipment': equipment}, f)

        frame_idx += 1

    cap.release()
    print(f"Processed {frame_idx} frames for equipment detection.")


if __name__ == "__main__":
    detect_equipment()
