import time
from ultralytics import YOLO
from utils import open_video, get_device, compute_fps

# ==============================
# CONFIGURAZIONE
# ==============================

VIDEO_PATH = "data/videos/input.mp4"

# Modello leggero
MODEL_PATH = "yolov8n.pt"

# Modelli più precisi (e più lenti)
#MODEL_PATH = "yolov8s.pt"
#MODEL_PATH = "yolov8m.pt"
# MODEL_PATH = "yolov8l.pt"


# SETUP


device = get_device()
print(f"Usando device: {device}")

model = YOLO(MODEL_PATH).to(device)

cap, width, height, fps = open_video(VIDEO_PATH)

print(f"Video: {width}x{height} @ {fps} FPS")


# BENCHMARK FPS


frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    _ = model(frame)  # inferenza (senza plot / output video)
    frame_count += 1

    if frame_count % 30 == 0:
        current_fps = compute_fps(start_time, frame_count)
        print(f"Frame processed: {frame_count} | Average FPS: {current_fps:.2f}")

cap.release()
elapsed = time.time() - start_time
average_fps = frame_count / elapsed

print("\n=== BENCHMARK RESULTS ===")
print(f"Total frames: {frame_count}")
print(f"Total time: {elapsed:.2f} s")
print(f"Average FPS: {average_fps:.2f}")