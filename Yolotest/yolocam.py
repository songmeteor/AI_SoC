import torch
import cv2

def nothing(x):
    pass

def main():
    # 모델 로드 (로컬 yolov5 폴더에서)
    model_path = "yolov5n.pt"
    model = torch.hub.load('/home/Pi/Work/yolov5', 'custom', path=model_path, source='local')
    model.eval()

    names = model.names

    # 웹캠 열기 (0 = 기본 카메라)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    window_name = "YOLOv5 Webcam Detection"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Confidence", window_name, 25, 100, nothing)  # 초기 0.25

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            break

        conf_thres = cv2.getTrackbarPos("Confidence", window_name) / 100.0

        # 추론
        results = model(frame, size=640)
        detections = results.xyxy[0]  # tensor (x1,y1,x2,y2,conf,cls)

        # 필터링 및 박스 그리기
        img = frame.copy()
        for *xyxy, conf, cls in detections:
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        # 화면 크기 조절 (필요시)
        scale = 0.75
        resized_img = cv2.resize(img, (0,0), fx=scale, fy=scale)

        cv2.imshow(window_name, resized_img)

        if cv2.waitKey(1) == 27:  # ESC 키
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()