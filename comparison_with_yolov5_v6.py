import cv2
import pafy

from YOLOv7 import YOLOv7
from YOLOv7.utils import draw_comparison

# Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://youtu.be/zPre8MgmcHY'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
start_time = 0  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * 60)

# Initialize object localizer
yolov7_path = "models/yolov7-tiny_736x1280.onnx"
yolov7_detector = YOLOv7(yolov7_path, conf_thres=0.5, iou_thres=0.5)

# Replace the model path with the YOLOv5 or YOLOv6 model path
# yolov5_v6_path = "models/yolov6s.onnx"
yolov5_v6_path = "models/yolov5s6.onnx"
yolov5_v6_model_name = "YOLOv6s" if "v6" in yolov5_v6_path else "YOLOv5s6"
yolov5_v6_detector = YOLOv7(yolov5_v6_path, conf_thres=0.5, iou_thres=0.5)

# out = cv2.VideoWriter('output3.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 720))

cv2.namedWindow("Model comparison", cv2.WINDOW_NORMAL)
while cap.isOpened():

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break
    except Exception as e:
        print(e)
        break

    # Update object localizer
    yolov7_detector(frame)

    yolov5_v6_detector(frame)

    yolov5_v6_img = yolov5_v6_detector.draw_detections(frame)
    yolov7_img = yolov7_detector.draw_detections(frame)

    size = 2.6
    text_thickness = 3
    combined_img = draw_comparison(yolov5_v6_img, yolov7_img, yolov5_v6_model_name, "YOLOV7-tiny", size, text_thickness)
    # out.write(combined_img)

    cv2.imshow("Model comparison", combined_img)

# out.release()