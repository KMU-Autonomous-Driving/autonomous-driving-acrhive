import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 사용할 클래스 ID (COCO 데이터셋 기준)
TARGET_CLASSES = [0, 2, 5, 7, 12]  # 사람, 자동차, 버스, 트럭, 도로 표지판

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # YOLOv8 모델 사용

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

def predict_and_detect(chosen_model, img, target_classes, conf=0.5):
    results = chosen_model(img, conf=conf)

    for result in results:
        for box in result.boxes:
            # 감지된 객체의 클래스 ID
            class_id = int(box.cls[0])
            
            # 필요한 클래스만 필터링
            if class_id in target_classes:
                # 바운딩 박스 좌표 및 라벨 표시
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
                cv2.putText(img, f"{result.names[class_id]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Process the image
        result_img = predict_and_detect(model, color_image, TARGET_CLASSES, conf=0.5)

        # Show images
        cv2.imshow("RealSense YOLOv8 Detection", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
