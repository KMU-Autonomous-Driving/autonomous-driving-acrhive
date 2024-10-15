from ultralytics import YOLO
import cv2
# 사용할 클래스 ID (COCO 데이터셋 기준)
TARGET_CLASSES = [0, 2, 5, 7, 12]  # 사람, 자동차, 버스, 트럭, 도로 표지판

# YOLOv8 모델 로드
model = YOLO('yolov10s.pt')  # YOLOv10s 모델 사용

# 비디오나 웹캠에서 실시간 객체 감지를 원할 경우
cap = cv2.VideoCapture(0)  # 웹캠 사용

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 객체 감지 수행
    results = model(frame)

    # 결과에서 필요한 클래스만 필터링
    annotated_frame = frame.copy()  # 원본 프레임 복사
    for result in results:
        for box in result.boxes:
            # 감지된 객체의 클래스 ID
            class_id = int(box.cls[0])
            
            # 필요한 클래스만 필터링
            if class_id in TARGET_CLASSES:
                # 바운딩 박스 좌표 및 라벨 표시
                annotated_frame = result.plot()  # 필요한 객체만 표시

    # 화면에 출력
    cv2.imshow('YOLOv10 Detection', annotated_frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
