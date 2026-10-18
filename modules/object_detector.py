from ultralytics import YOLO
import cv2
import numpy as np
import random

# پالت رنگ ثابت برای هر کلاس (تا باکس‌ها بین فریم‌ها ثابت بمانند)
_rng = random.Random(42)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
CLASS_COLORS = {name: tuple(_rng.randint(60, 255) for _ in range(3)) for name in COCO_CLASSES}


class ObjectDetector:
    """YOLOv8 wrapper با رسم سفارشی، فیلتر کلاس و آمار."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = YOLO(model_name)
        self.names = self.model.names  # {id: name}

    # ---------- Utilities ----------
    @staticmethod
    def _draw_label(img, text, x, y, color):
        """برچسب با پس‌زمینه رنگی خوانا."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, th + 4)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
        cv2.putText(img, text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- Core ----------
    def detect(self, image_bgr, conf=0.4, iou=0.5, classes_filter=None):
        """
        image_bgr: np.ndarray BGR
        classes_filter: list[int] | None  → فقط این کلاس‌ها نمایش داده می‌شوند
        returns: (annotated, detections_list, stats)
        """
        results = self.model(image_bgr, conf=conf, iou=iou, verbose=False)[0]
        annotated = image_bgr.copy()
        detections = []
        class_counts = {}

        for box in results.boxes:
            cls_id = int(box.cls)
            if classes_filter and cls_id not in classes_filter:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_v = float(box.conf)
            cls_name = self.names[cls_id]
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))

            # باکس و برچسب
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            self._draw_label(annotated, f"{cls_name} {conf_v:.2f}", x1, y1, color)

            detections.append({
                "class": cls_name,
                "class_id": cls_id,
                "confidence": round(conf_v, 3),
                "bbox_xyxy": [x1, y1, x2, y2],
                "width": x2 - x1,
                "height": y2 - y1,
            })
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        stats = {
            "total": len(detections),
            "by_class": class_counts,
        }
        return annotated, detections, stats
