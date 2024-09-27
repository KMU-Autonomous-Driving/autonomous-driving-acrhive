from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드 (사전 학습된 모델 사용)
model = YOLO('yolov8n.pt')

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, results

# 이미지 읽기
image = cv2.imread("YourImagePath.png")

result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

cv2.imshow("Image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 비디오 처리
video_path = r"YourVideoPath"
cap = cv2.VideoCapture(video_path)

# 비디오 작성기 생성 함수
def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    return writer

output_filename = "YourOutputFilename.mp4"
writer = create_video_writer(cap, output_filename)

while cap.isOpened():
    success, img = cap.read()

    if not success:
        break

    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    writer.write(result_img)
    cv2.imshow("Video", result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()


