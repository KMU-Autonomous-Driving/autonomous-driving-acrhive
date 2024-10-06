from ultralytics import YOLO
import cv2


model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened() :
    ret, frame = cap.read()

    if ret :
        results = model(frame)
        cv2.imshow("Results", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()