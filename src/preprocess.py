import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple

def preprocess_video(
    video_path: str,
    target_fps: int = 24,
    output_size: Tuple[int, int] = (256, 256),
    center_crop: bool = False
) -> np.ndarray:
    """
    Load video, resample to `target_fps`, resize frames to `output_size`,
    convert BGR->RGB, normalize to [0,1], and return as an ndarray.

    Args:
        video_path (str): Path to input .mp4 file.
        target_fps (int): Desired frame rate for sampling.
        output_size (Tuple[int,int]): (height, width) of output frames.
        center_crop (bool): If True, center-crop before resizing to maintain aspect ratio.

    Returns:
        frames (np.ndarray): Float32 array of shape (T, H, W, 3), values in [0,1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    # Original video FPS
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = target_fps

    frame_interval = max(int(round(orig_fps / target_fps)), 1)

    processed_frames = []
    index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample at approximate target_fps
        if index % frame_interval == 0:
            # Optional center crop to square
            if center_crop:
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                frame = frame[top:top+min_dim, left:left+min_dim]

            # Resize to output_size
            frame = cv2.resize(frame, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Normalize to [0,1]
            # frame = frame.astype(np.float32) / 255.0  #Might cause problems with certain models

            processed_frames.append(frame)

        index += 1

    cap.release()

    if not processed_frames:
        raise ValueError("No frames were processed; check video length or FPS settings.")

    # Stack to a single tensor
    return np.stack(processed_frames, axis=0)

def save_processed_video(
    frames: np.ndarray,
    output_path: str,
    fps: int = 30,
    codec: str = "mp4v"
) -> None:
    """
    Save a preprocessed video tensor to disk.

    Args:
        frames (np.ndarray): Array of shape (T, H, W, 3), dtype float32 in [0,1].
        output_path (str): Where to write the .mp4 (or .avi) file.
        fps (int): Frame rate to encode at.
        codec (str): FourCC code, e.g. "mp4v", "XVID", "H264".
    """
    # Determine frame size
    T, H, W, C = frames.shape
    assert C == 3, "Expected 3 color channels"

    # Convert normalized floats back to uint8 BGR
    # (VideoWriter expects BGR)
    # frames_uint8 = (frames * 255.0).clip(0, 255).astype(np.uint8)
    frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]

    # Set up writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for {output_path}")

    # Write out each frame
    for frame in frames_bgr:
        writer.write(frame)
    writer.release()

if __name__ == "__main__":
    video_path = "data\\raw\\Video-1.mp4"
    frames = preprocess_video(video_path)
    save_processed_video(frames, "data\\processed\\preprocessed_video-1.mp4", fps=24, codec="mp4v")
    print(f"Processed {len(frames)} frames and saved to data\\processed\\preprocessed_video-1.mp4")