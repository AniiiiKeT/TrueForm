import os
import cv2
import logging
import warnings
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG for detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/keypoint_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KeypointExtractor:
    def __init__(self, video_path, output_path, save_keypoints=False):
        self.base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task')
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            output_segmentation_masks=True,
            running_mode=self.VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.95,
        )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)

        self.video_path = video_path
        self.output_path = output_path
        self.file_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.save_keypoints = save_keypoints

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        return annotated_image

    def extractor(self):
        logger.info(f"Opening video file: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {self.video_path}")
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Video properties: {fps:.2f} FPS, {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_file = os.path.join(self.output_path, self.file_name + ".mp4")
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        keypoints = []
        frame_idx = 0

        while True:
            success, frame_bgr = cap.read()
            if not success:
                logger.info("End of video reached.")
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = self.detector.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                logger.warning(f"No pose landmarks detected in frame {frame_idx}")
                continue

            lm = result.pose_landmarks[0]
            coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            keypoints.append(coords)

            annotated_rgb = self.draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            out.write(annotated_bgr)

            if frame_idx % 50 == 0:
                logger.info(f"Processed {frame_idx} frames...")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        logger.info(f"Saved annotated video to {output_file}")

        kp_array = None
        if self.save_keypoints and keypoints:
            kp_array = np.stack(keypoints, axis=0)
            os.makedirs("data/keypoints", exist_ok=True)
            np.save(f"data/keypoints/{self.file_name}_landmarks.npy", kp_array)
            logger.info(f"Saved landmarks array to data/keypoints/{self.file_name}_landmarks.npy")

        return kp_array

if __name__ == "__main__":
    video_path = "D:/Projects/TrueForm/data/raw/video_0.mp4"
    output_path = "D:/Projects/TrueForm/outputs"

    extractor = KeypointExtractor(video_path, output_path, save_keypoints=True)
    keypoints = extractor.extractor()

    if keypoints is not None:
        logger.info(f"Extracted {len(keypoints)} frames of keypoints.")
    else:
        logger.warning("No keypoints extracted.")
