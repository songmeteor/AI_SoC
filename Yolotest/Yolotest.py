import torch
import cv2
import numpy as np
import os

# 모델 로드
model_path = "yolov5n.pt"
model = torch.hub.load('/home/Pi/Work/yolov5', 'custom', path=model_path, source='local')
model.eval()

# 클래스 이름 로딩
names = model.names

# 슬라이더 콜백 함수 (필수이지만 내용 없음)
def nothing(x):
    pass

def main():
    # 이미지 경로 직접 지정
    image_path = "/home/Pi/Work/yolov5/image/test1.jpg"
    
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("이미지를 불러올 수 없습니다:", image_path)
        return

    window_name = "YOLOv5 Detection"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Confidence", window_name, 25, 100, nothing)  # 초기값 25%

    while True:
        img = orig_img.copy()
        conf_thres = cv2.getTrackbarPos("Confidence", window_name) / 100.0

        results = model(img, size=640)
        detections = results.xyxy[0]  # tensor: (x1, y1, x2, y2, conf, cls)

        for *xyxy, conf, cls in detections:
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # 이미지 축소 출력
        scale = 0.5
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        cv2.imshow(window_name, resized_img)

        if cv2.waitKey(30) == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()