import pyrealsense as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

def grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def canny(image):
    return cv2.Canny(image, 50, 150)

def region(image):
    height, width = image.shape
    triangle = np.array([
        [(100, height), (475, 325), (width, height)]
    ])
    
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def average(image, lines):
    left = []
    right = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_int = parameters[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    if len(left) == 0 or len(right) == 0:
        return None

    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line]) 

def make_points(image, average): 
    if average is None:
        return None
    slope, y_int = average 
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image

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
        copy = np.copy(color_image)
        grey_image = grey(copy)
        gaus_image = gauss(grey_image)
        edges = canny(gaus_image)
        isolated = region(edges)
        lines = cv2.HoughLinesP(isolated, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=50)
        averaged_lines = average(copy, lines)
        black_lines = display_lines(copy, averaged_lines)
        lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)

        # Show images
        cv2.imshow('RealSense', lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
