from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('runs/detect/football_player_model_3/weights/best.pt')
results = model.predict('sample_images/image.jpg')
result = results[0]

img = result.orig_img
names = result.names
scores = result.boxes.conf.numpy()
classes = result.boxes.cls.numpy()
boxes = result.boxes.xyxy.numpy().astype(np.int32)

for box, score, cls in zip(boxes, scores, classes):
    class_label = names[cls]
    label = f"{class_label} : {score:0.2f}"
    lbl_margin = 3
    img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=1)
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    lbl_w, lbl_h = label_size[0]
    lbl_w += 2 * lbl_margin
    lbl_h += 2 * lbl_margin

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()