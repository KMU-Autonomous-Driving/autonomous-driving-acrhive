import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

if __name__ == "__main__":
    # Load model
    model = YOLO("best.pt")
    #model is based on a Hong Kong traffic sign

    # Open webcam
    cap = cv2.VideoCapture(0)

    found_class = set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict on the frame
        results = model.predict(frame)

        annotator = Annotator(frame)
        boxes = results[0].boxes
        for box in boxes:
            # Get box coordinates in (top, left, bottom, right) format
            coordinates = box.xyxy[0]
            class_id = box.cls
            class_name = "-".join(model.names[int(class_id)].split("--")[1:-1])
            found_class.add(class_name)
            annotator.box_label(coordinates, class_name)

        frame = annotator.result()
        cv2.imshow('YOLO V8 Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(found_class)
