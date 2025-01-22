from ultralytics import YOLO

model = YOLO('runs/detect/football_player_model_3/weights/best.pt')
results = model.val(data='data.yaml', plots=True)

# Print evaluation metrics
print(f"Validation Results: {results}")

# Optionally, you can access more detailed metrics like mAP, losses, etc.
# For example:
print(f"mAP@0.5: {results.box.map50}")
print(f"mAP@0.5:0.95: {results.box.map}")