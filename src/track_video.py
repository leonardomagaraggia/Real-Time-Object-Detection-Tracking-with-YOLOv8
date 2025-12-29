from ultralytics import YOLO
from utils import get_device

# CONFIGURAZIONE MODELLO

# Modello leggero (default, più veloce)
MODEL_PATH = "yolov8n.pt"

# Modelli più pesanti e precisi
# MODEL_PATH = "yolov8s.pt"   # small
# MODEL_PATH = "yolov8m.pt"   # medium
# MODEL_PATH = "yolov8l.pt"   # large (molto lento su CPU)

VIDEO_PATH = "data/videos/input.mp4"


# SETUP

device = get_device()
print(f"Usando device: {device}")

model = YOLO(MODEL_PATH).to(device)


# TRACKING


model.track(
    source=VIDEO_PATH,
    tracker="bytetrack.yaml",
    save=True,
    project="data/output",
    name="tracking",
    conf=0.4,
    iou=0.5,
    show=False
)

print("Tracking completed")
print("Output saved in: data/output/tracking/")