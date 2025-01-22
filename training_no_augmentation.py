import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

data_path = 'data.yaml'
model_path = 'yolov8n.pt'

model = YOLO(model_path)

model.train(
    data=data_path, epochs=5, batch=16, imgsz=640, lr0=0.00001, hsv_h=0.0,
    hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
    perspective=0.0, flipud=0.0, fliplr=0.0, bgr=0.0, mosaic=0.0, mixup=0.0, 
    copy_paste=0.0, erasing=0.0, crop_fraction=0, augment=False,
    name='football_player_model_no_augmentation', exist_ok=True
)