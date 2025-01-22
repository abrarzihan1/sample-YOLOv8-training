from ultralytics import YOLO
import numpy as np
import cv2

model = YOLO('runs/detect/football_player_model_no_augmentation/weights/best.pt')
cap = cv2.VideoCapture('sample_videos/football_game.mp4')

threshold = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 332))

    if not ret:
        print("Video not returned")
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        color = (class_id*10, class_id*100, class_id*10)

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3, cv2.LINE_AA)

    cv2.imshow("frame", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()