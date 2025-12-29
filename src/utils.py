import cv2
import time
import os
import torch

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    return cap, width, height, fps


def create_video_writer(output_path, width, height, fps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )


def compute_fps(start_time, frame_count):
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
