import cv2
from PIL import Image
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO('best_traffic_small_yolo.pt')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())

        # Run inference on the current frame
        results = model(frame)  # results list

        for r in results:
            frame = r.plot()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Press 'q' on keyboard to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
