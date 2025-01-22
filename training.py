import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

data_path = 'data.yaml'
model_path = 'yolov8n.pt'

model = YOLO(model_path)

model.train(
    data=data_path,
    epochs=10,
    batch=16,
    imgsz=640,
    name='football_player_model_3',
    exist_ok=True
)