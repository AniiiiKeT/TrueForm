import os
import cv2
import logging
import warnings
import math
import numpy as np
import pandas as pd
from mediapipe import solutions

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/annotate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoAnnotator():
    def __init__(self, video_path, output_path, landmarks, metrics):
        self.video_path = video_path
        self.output_path = output_path
        self.output_file_name = os.path.join(output_path, "annotated_" + os.path.basename(video_path))
        self.landmarks = landmarks
        self.metrics = metrics

    def annotate(self):

        TH_SYMMETRY_DEG = 5.0     # degrees: peak left/right elbow difference
        TH_HOLD_VAR     = 2.0     # degrees: stddev of hold angle
        TH_STANCE_RATIO_MIN = 0.8 # stance_width / shoulder_span < -> too narrow
        TH_STANCE_RATIO_MAX = 1.5 # > -> too wide
        TH_RELEASE_JERK = 10.0    # degrees/frame: big change right after release
        HOLD_WINDOW_FRAMES = 10   # frames before peak considered "hold"

        L = solutions.pose.PoseLandmark

        def dist(a, b):
            return np.linalg.norm(a - b)

        def angle_between(a, b, c):
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
            if denom == 0:
                return 0.0
            cosang = np.dot(ba, bc) / denom
            return math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))

        def normalized_to_pixel(norm_pt, width, height):
            # norm_pt is (x,y,z)
            return int(norm_pt[0] * width), int(norm_pt[1] * height)

        def draw_text_box(img, text_lines, origin=(10,30), font_scale=0.45, thickness=1, max_width_ratio=0.45):
            """
            Draws a semi-transparent text box with word wrapping and optional small font size.
            """
            x0, y0 = origin
            font = cv2.FONT_HERSHEY_SIMPLEX
            max_width_px = int(img.shape[1] * max_width_ratio)

            wrapped_lines = []
            for line in text_lines:
                words = line.split(" ")
                current_line = ""
                for word in words:
                    test_line = (current_line + " " + word).strip()
                    (tw, th), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                    if tw <= max_width_px:
                        current_line = test_line
                    else:
                        wrapped_lines.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_lines.append(current_line)

            # Compute box size
            w = 0
            h = 0
            for line in wrapped_lines:
                (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
                w = max(w, tw)
                h += th + 6

            # Background rectangle
            cv2.rectangle(img, (x0-6, y0-16), (x0+w+6, y0 + h), (0,0,0), -1)

            # Put text
            y = y0
            for line in wrapped_lines:
                cv2.putText(img, line, (x0, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
                (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
                y += th + 6

        df = self.metrics
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Can't open video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file_name, fourcc, fps, (W, H))

        # Precompute some per-shot diagnostics (more could be added)
        diagnostics = {}  # shot_idx -> dict

        for shot_idx, row in df.iterrows():
            peak = int(row['frame_peak'])
            start = int(row['frame_start'])
            # shoulder span at peak (normalized)
            coords_peak = self.landmarks[peak]
            shoulder_span = dist(coords_peak[L.LEFT_SHOULDER.value], coords_peak[L.RIGHT_SHOULDER.value])
            # stance width already in df (assumed normalized)
            stance_w = row.get('stance_width', None)
            stance_ratio = None
            if stance_w is not None and shoulder_span > 0:
                stance_ratio = stance_w / shoulder_span

            # hold variance and symmetry already present in df
            hold_var = row.get('hold_variance', 0.0)
            symmetry = row.get('symmetry_diff', 0.0)

            # Release jerk: compute change in elbow angle immediately after peak
            left_angles = []
            for f in range(peak, min(peak+4, self.landmarks.shape[0])):
                c = self.landmarks[f]
                a = angle_between(c[L.LEFT_SHOULDER.value], c[L.LEFT_ELBOW.value], c[L.LEFT_WRIST.value])
                left_angles.append(a)
            release_jerk = 0.0
            if len(left_angles) >= 2:
                release_jerk = max(abs(left_angles[i+1] - left_angles[i]) for i in range(len(left_angles)-1))
            # head stability: stddev of nose during hold window
            hold_start = max(start, peak - HOLD_WINDOW_FRAMES)
            nose_positions = self.landmarks[hold_start:peak, L.NOSE.value] if peak > hold_start else np.array([])
            head_std = 0.0
            if len(nose_positions) > 0:
                # compute pixel variance in y as proxy for vertical stability
                head_std = float(np.std(nose_positions[:,1]))

            diagnostics[shot_idx] = {
                "shoulder_span": shoulder_span,
                "stance_w": stance_w,
                "stance_ratio": stance_ratio,
                "hold_var": float(hold_var),
                "symmetry": float(symmetry),
                "release_jerk": float(release_jerk),
                "head_std": float(head_std),
                "frame_start": start,
                "frame_peak": peak
            }


        frame_idx = 0
        shots_by_frame = {}
        for shot_idx, diag in diagnostics.items():
            s = int(diag['frame_start'])
            p = int(diag['frame_peak'])
            # annotate from start to (peak + 10) frames to include release
            end = min(p + 10, self.landmarks.shape[0]-1)
            for f in range(s, end+1):
                shots_by_frame[f] = shot_idx

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            annotated = frame.copy()

            if frame_idx < len(self.landmarks):
                lm_norm = self.landmarks[frame_idx]  # (33,3)
            else:
                lm_norm = None

            pix = {}
            if lm_norm is not None:
                # convert normalized to pixels for all landmarks (used for drawing always)
                for i in range(lm_norm.shape[0]):
                    x_px = int(lm_norm[i,0] * w)
                    y_px = int(lm_norm[i,1] * h)
                    pix[i] = (x_px, y_px)

                # DRAW SKELETON (always visible when landmarks exist)
                # bones in neutral gray
                for a, b in solutions.pose.POSE_CONNECTIONS:
                    pa = pix[int(a)]
                    pb = pix[int(b)]
                    cv2.line(annotated, pa, pb, (180, 180, 180), 2)

                # default joint color (neutral)
                for i, (x, y) in pix.items():
                    cv2.circle(annotated, (x, y), 5, (200, 200, 200), -1)

                # annotate left/right shoulders for orientation clarity (always visible)
                for i in (L.LEFT_SHOULDER.value, L.RIGHT_SHOULDER.value):
                    if i in pix:
                        x, y = pix[i]
                        cv2.putText(annotated, 'L-S' if i==L.LEFT_SHOULDER.value else 'R-S',
                                    (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1, cv2.LINE_AA)

                # Draw an arrow showing target direction (optional): use vector from hips to nose as rough facing
                hip_mid = ((lm_norm[L.LEFT_HIP.value] + lm_norm[L.RIGHT_HIP.value]) / 2.0)
                nose = lm_norm[L.NOSE.value]
                hip_px = normalized_to_pixel(hip_mid, w, h)
                nose_px = normalized_to_pixel(nose, w, h)
                cv2.arrowedLine(annotated, hip_px, nose_px, (255,200,0), 2, tipLength=0.15)

            # If this frame is within a shot, overlay suggestions and highlight bad joints
            if frame_idx in shots_by_frame and lm_norm is not None:
                shot_idx = shots_by_frame[frame_idx]
                diag = diagnostics[shot_idx]
                s = diag['frame_start']
                p = diag['frame_peak']

                # compute suggestion list
                suggestions = []
                bad_joints = []

                if diag['symmetry'] > TH_SYMMETRY_DEG:
                    suggestions.append(f"Asymmetric draw: {diag['symmetry']:.1f} degrees between arms")
                    bad_joints += [L.LEFT_ELBOW.value, L.RIGHT_ELBOW.value]
                if diag['hold_var'] > TH_HOLD_VAR:
                    suggestions.append("Unstable anchor: hold variance high")
                    bad_joints += [L.LEFT_WRIST.value, L.RIGHT_WRIST.value]
                if diag['stance_ratio'] is not None:
                    if diag['stance_ratio'] < TH_STANCE_RATIO_MIN:
                        suggestions.append("Stance too narrow: widen feet")
                        bad_joints += [L.LEFT_ANKLE.value, L.RIGHT_ANKLE.value]
                    elif diag['stance_ratio'] > TH_STANCE_RATIO_MAX:
                        suggestions.append("Stance too wide: bring feet closer")
                        bad_joints += [L.LEFT_ANKLE.value, L.RIGHT_ANKLE.value]
                if diag['release_jerk'] > TH_RELEASE_JERK:
                    suggestions.append("Jerky release detected: work on smooth follow-through")
                    bad_joints += [L.LEFT_WRIST.value]
                if diag['head_std'] > 0.02:  # normalized pos variance threshold (tune)
                    suggestions.append("Head movement high during hold: stabilize head")
                    bad_joints += [L.NOSE.value]

                if len(suggestions) == 0:
                    suggestions = ["No major issues detected — good draw!"]

                # Re-draw joints with shot-specific coloring (green by default, red for bad joints)
                for i, (x, y) in pix.items():
                    color = (0,255,0) if i not in bad_joints else (0,0,255)
                    cv2.circle(annotated, (x,y), 6, color, -1)

                # Overwrite bones with a slightly brighter color so shot frames stand out
                for a, b in solutions.pose.POSE_CONNECTIONS:
                    pa = pix[int(a)]
                    pb = pix[int(b)]
                    cv2.line(annotated, pa, pb, (200,200,200), 2)

                # Put suggestions text box
                draw_text_box(annotated, suggestions, origin=(10,30))

                # Draw shot label/metadata
                cv2.putText(annotated, f"Shot {shot_idx} Frame {frame_idx}", (10, H-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # Optionally draw frame index
            cv2.putText(annotated, f"Frame {frame_idx}", (W-160, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            out.write(annotated)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"Saved annotated video to: {self.output_path}")


if __name__ == "__main__":
    from extract_keypoints import KeypointExtractor
    from shot_metrics import ShotMetrics

    video_path = "D:/Projects/TrueForm/data/raw/video_0.mp4"
    output_path = "D:/Projects/TrueForm/outputs"

    keypoint_extractor = KeypointExtractor(video_path, output_path)
    landmarks = keypoint_extractor.extractor()
    file_name = "video_1"
    fps = 30

    shot_metrics = ShotMetrics(landmarks, fps, file_name)
    metrics = shot_metrics.get_shot_metrics()

    annotator = VideoAnnotator(video_path, output_path, landmarks, metrics)
    annotator.annotate()