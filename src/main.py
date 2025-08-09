from extract_keypoints import KeypointExtractor
from shot_metrics import ShotMetrics
from annotate import VideoAnnotator

import os

class Main():
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.fps = 30
        self.file_name = os.path.splitext(os.path.basename(video_path))[0]

    def run(self):
        extractor = KeypointExtractor(self.video_path, self.output_path, save_keypoints=True)
        keypoints = extractor.extractor()
        self.fps = extractor.fps

        shot_metrics = ShotMetrics(keypoints, self.fps, self.file_name, save=True)
        metrics = shot_metrics.get_shot_metrics()

        video_annotator = VideoAnnotator(self.video_path, self.output_path, keypoints, metrics)
        video_annotator.annotate()
        print("Done!")


for files in os.listdir("data/raw"):
    if files.endswith(".mp4"):
        video_path = os.path.join("data/raw", files)
        output_path = "outputs"

        main = Main(video_path, output_path)

        print(f"Processing video: {os.path.basename(video_path)}")
        main.run()

