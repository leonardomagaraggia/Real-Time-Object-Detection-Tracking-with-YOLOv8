from ultralytics import YOLO
import time
from utils import open_video, create_video_writer, compute_fps, get_device

# CONFIGURAZIONE MODELLO

# Modello leggero (default, più veloce)
MODEL_PATH = "yolov8n.pt"

# Modelli più pesanti e precisi
# MODEL_PATH = "yolov8s.pt"   # small
# MODEL_PATH = "yolov8m.pt"   # medium
# MODEL_PATH = "yolov8l.pt"   # large (molto lento su CPU)

VIDEO_PATH = "data/videos/input.mp4"
OUTPUT_PATH = "data/output/detection.mp4"

# SETUP


device = get_device()
print(f"Usando device: {device}")

model = YOLO(MODEL_PATH).to(device)

cap, width, height, fps = open_video(VIDEO_PATH)
out = create_video_writer(OUTPUT_PATH, width, height, fps)


# INFERENCE LOOP

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    frame_count += 1

    if frame_count % 30 == 0:
        current_fps = compute_fps(start_time, frame_count)
        print(f"Frame: {frame_count} | FPS avg: {current_fps:.2f}")

# CLEANUP


cap.release()
out.release()

total_fps = compute_fps(start_time, frame_count)
print(f"Detection completed")
print(f"Total frames: {frame_count}")
print(f"FPS avg final: {total_fps:.2f}")
print(f"Output saved in: {OUTPUT_PATH}")