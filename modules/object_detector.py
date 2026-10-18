from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, image_bgr, conf=0.4, iou=0.5):
        results = self.model(image_bgr, conf=conf, iou=iou, verbose=False)
        annotated = results[0].plot()
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": self.model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })
        return annotated, detections