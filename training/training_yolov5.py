# using YOLOv5 as it provides better ball tracking
from ultralytics import YOLO
import os
import time

# get path
base_dir = os.path.dirname(__file__)  
data_path = os.path.join(base_dir, "football-players-detection-1", "data.yaml")

model = YOLO("yolov5su.pt")

start = time.time()
result = model.train(epochs=100, patience=10, data=data_path)
end = time.time()

print(f"Training duration: {end - start}")