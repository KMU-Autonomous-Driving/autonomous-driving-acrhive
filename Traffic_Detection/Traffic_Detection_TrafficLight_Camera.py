# import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")

# 클래스 이름 로드
classes = model.names

# 웹캠 설정
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv8을 사용하여 교통 신호등 감지
        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = box.conf
                if confidence > 0.5 and class_id == 9:  # class_id 9는 교통 신호등
                    x1, y1, x2, y2 = map(int, box.xyxy)
                    label = str(classes[class_id])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Traffic Light Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()


# # RealSense 카메라 설정
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

# try:
#     while True:
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue

#         color_image = np.asanyarray(color_frame.get_data())

#         # YOLOv8을 사용하여 교통 신호등 감지
#         results = model(color_image)

#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 class_id = int(box.cls)
#                 confidence = box.conf
#                 if confidence > 0.5 and class_id == 9:  # class_id 9는 교통 신호등
#                     x1, y1, x2, y2 = map(int, box.xyxy)
#                     label = str(classes[class_id])
#                     color = (0, 255, 0)
#                     cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(color_image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         cv2.imshow("Traffic Light Detection", color_image)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()
