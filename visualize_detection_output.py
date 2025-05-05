from ultralytics import YOLO
import cv2
import os


# class_names = [
#     '대형차_불법차량', '대형차_정상차량',
#     '중형차_불법차량', '중형차_정상차량',
#     '소형차_불법차량', '소형차_정상차량'
# ]

class_names = ['0', '1', 
               '2', '3', 
               '4', '5']

save_dir = 'saved_rois'
os.makedirs(save_dir, exist_ok=True)

model = YOLO("best.pt")
model.to('cpu')

video_path = "MBC_re.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
roi_count = 0  # 저장 ROI 이미지

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5)[0]
    for box in results.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if cls not in [0, 2, 4]:  # 불법 차량 클래스만 고려
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f'{class_names[cls]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ROI 이미지 자르고 저장
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            filename = f'{save_dir}/roi_{roi_count}.jpg'
            cv2.imwrite(filename, roi)
            roi_count += 1

    cv2.imshow('Detection Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()