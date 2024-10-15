import numpy as np
import matplotlib.pyplot as plt
import cv2
# import pyrealsense2 as rs

# 차량의 초기 위치와 각도 설정
x, y, theta = 0, 0, 0

# 차량의 이동 경로를 저장할 리스트
path_x, path_y = [x], [y]

# # RealSense 카메라 설정
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

# 웹캠 설정
cap = cv2.VideoCapture(0)

# # 주차 공간 인식 함수
def detect_parking_space(frame):
    # 주차 공간 인식을 위한 간단한 코드(실제 구현에서는 더 복잡한 알고리즘 필요)
    # 여기서는 단순히 이미지 중앙의 픽셀 값을 사용
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    target_x = center_x / 100.0  # 픽셀 값을 미터 단위로 변환 (예시)
    target_y = center_y / 100.0

    return target_x, target_y, frame

# # 주차 공간 인식 함수
# def detect_parking_space(depth_frame, color_frame):
#     # 깊이 프레임을 numpy 배열로 변환
#     depth_image = np.asanyarray(depth_frame.get_data())
#     color_image = np.asanyarray(color_frame.get_data())

#     # 주차 공간 인식을 위한 간단한 코드 (실제 구현에서는 더 복잡한 알고리즘 필요)
#     # 여기서는 단순히 이미지 중앙의 깊이 값을 사용
#     center_depth = depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2]
#     target_x = center_depth / 1000.0  # 깊이 값을 미터 단위로 변환
#     target_y = center_depth / 1000.0

#     return target_x, target_y, color_image

# 간단한 이동 함수 정의
def move(x, y, theta, distance, angle):
    x += distance * np.cos(theta)
    y += distance * np.sin(theta)
    theta += angle
    return x, y, theta

# 주차를 위한 경로 생성(웹캠)
for _ in range(100):
    ret, frame = cap.read()
    if not ret:
        continue
    
    target_x, target_y, color_image = detect_parking_space(frame)
    
    distance = 0.1
    angle = 0.01
    x, y, theta = move(x, y, theta, distance, angle)
    path_x.append(x)
    path_y.append(y)
    
    if np.hypot(target_x - x, target_y - y) < 0.1:
        break

# # 주차를 위한 경로 생성(카메라)
# for _ in range(100):
#     frames = pipeline.wait_for_frames()
#     depth_frame = frames.get_depth_frame()
#     color_frame = frames.get_color_frame()
    
#     if not depth_frame or not color_frame:
#         continue
    
#     target_x, target_y, color_image = detect_parking_space(depth_frame, color_frame)
    
#     distance = 0.1
#     angle = 0.01
#     x, y, theta = move(x, y, theta, distance, angle)
#     path_x.append(x)
#     path_y.append(y)
    
#     if np.hypot(target_x - x, target_y - y) < 0.1:
#         break

# 경로 시각화
plt.plot(path_x, path_y, label='Path')
plt.scatter(target_x, target_y, color='red', label='Target')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# 카메라 이미지 시각화
plt.imshow(color_image)
plt.title('Camera View')
plt.show()

# # RealSense 파이프라인 종료
# pipeline.stop()

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
