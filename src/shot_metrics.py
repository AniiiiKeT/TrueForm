import os
import cv2
import logging
import math
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe import solutions
from scipy.signal import find_peaks
from scipy.signal import savgol_filter

logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/shot_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ShotMetrics():
    def __init__(self, landmarks, fps, file_name, save = False):
        self.fps = fps
        self.save = save
        self.file_name = file_name
        self.landmarks = landmarks
        self.L = solutions.pose.PoseLandmark

    def angle_between(self, a, b, c):
        ba = a - b
        bc = c - b
        cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return math.degrees(math.acos(np.clip(cos_ang, -1.0, 1.0)))
    
    def dist(self, a, b):
        return np.linalg.norm(a - b)

    def elbow_angles(self):
        F = self.landmarks.shape[0]
        left_elbow = np.zeros(F)
        right_elbow = np.zeros(F)

        for i in range(F):
            coords = self.landmarks[i]
            left_elbow[i]  = self.angle_between(
                coords[self.L.LEFT_SHOULDER.value],
                coords[self.L.LEFT_ELBOW.value],
                coords[self.L.LEFT_WRIST.value]
            )
            right_elbow[i] = self.angle_between(
                coords[self.L.RIGHT_SHOULDER.value],
                coords[self.L.RIGHT_ELBOW.value],
                coords[self.L.RIGHT_WRIST.value]
            )

        window_length = 11
        polyorder     = 2

        left_smooth  = savgol_filter(left_elbow,  window_length, polyorder)
        right_smooth = savgol_filter(right_elbow, window_length, polyorder)

        return (left_smooth, right_smooth)

    def get_peak(self, signal):  
        peaks, props = find_peaks(signal, height=155, distance=60)
        logger.info(f"Detected {len(peaks)} shots at frames {peaks}")
        return peaks

    def get_shot_metrics(self):
        shot_metrics = []
        stance_widths = []

        (left_elbow_signal, right_elbow_signal) = self.elbow_angles()

        peaks_l = self.get_peak(left_elbow_signal)
        peaks_r = self.get_peak(right_elbow_signal)

        for p in peaks_l:
            start = np.argmin(left_elbow_signal[max(0, p-50):p]) + max(0, p-50)
            peak_angle = left_elbow_signal[p]
            hold_var   = np.std(left_elbow_signal[p-10:p])
            pull_rate  = (peak_angle - left_elbow_signal[start]) / (p - start)
            shot_metrics.append({
                "frame_start": start,
                "frame_peak" : p,
                "peak_angle" : peak_angle,
                "hold_variance": hold_var,
                "pull_rate": pull_rate,
            })
        df = pd.DataFrame(shot_metrics)
        df.index.name = "shot_idx"

        df['peak_angle_R'] = right_elbow_signal[df['frame_peak']]
        df['symmetry_diff'] = (df['peak_angle'] - df['peak_angle_R']).abs()
        df['duration_s'] = (df['frame_peak'] - df['frame_start']) / self.fps
        df['consistency_score'] = 1.0 / (1.0 + df['hold_variance'])

        for _, row in df.iterrows():
            coords = self.landmarks[int(row['frame_peak'])]
            lw = coords[self.L.LEFT_ANKLE.value]
            rw = coords[self.L.RIGHT_ANKLE.value]
            stance_widths.append(self.dist(lw, rw))

        df['stance_width'] = stance_widths

        if self.save:
            os.makedirs("data/shot_metrics", exist_ok=True)
            df.to_csv(f"data/metrics/{self.file_name}_shot_metrics.csv", index=False)
            logger.info(f"Saved shot metrics to data/metrics/{self.file_name}_shot_metrics.csv")
        
        return df

if __name__ == "__main__":

    from extract_keypoints import KeypointExtractor

    keypoint_extractor = KeypointExtractor("D:/Projects/TrueForm/data/raw/video_0.mp4", "D:/Projects/TrueForm/outputs")
    landmarks = keypoint_extractor.extractor()
    file_name = "video_0"
    fps = 30

    shot_metrics = ShotMetrics(landmarks, fps, file_name, save=True)
    print(type(shot_metrics.get_shot_metrics()))