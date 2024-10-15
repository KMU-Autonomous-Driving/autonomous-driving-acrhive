import cv2 as cv
import numpy as np
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')

# 웹캠에서 이미지 캡처 및 처리
cap = cv.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8 모델을 사용하여 객체 검출
        results = model(frame)

        # 검출된 객체에 바운딩 박스 그리기
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(class_id)]}: {confidence:.2f}"
                color = (0, 255, 0)
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 결과 출력
        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv.destroyAllWindows()

# import cv2 as cv
# import numpy as np
# import pyrealsense2 as rs
# from ultralytics import YOLO

# # YOLOv8 모델 로드
# model = YOLO('yolov8n.pt')

# # RealSense 카메라 설정
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# # 스트림 시작
# pipeline.start(config)

# try:
#     while True:
#         # 프레임 캡처
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue

#         # 이미지를 numpy 배열로 변환
#         frame = np.asanyarray(color_frame.get_data())

#         # YOLOv8 모델을 사용하여 객체 검출
#         results = model(frame)

#         # 검출된 객체에 바운딩 박스 그리기
#         for result in results:
#             boxes = result.boxes.xyxy.cpu().numpy()
#             confidences = result.boxes.conf.cpu().numpy()
#             class_ids = result.boxes.cls.cpu().numpy()

#             for box, confidence, class_id in zip(boxes, confidences, class_ids):
#                 x1, y1, x2, y2 = map(int, box)
#                 label = f"{model.names[int(class_id)]}: {confidence:.2f}"
#                 color = (0, 255, 0)
#                 cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # 결과 출력
#         cv.imshow('RealSense', frame)
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     # 스트림 종료
#     pipeline.stop()
#     cv.destroyAllWindows()
