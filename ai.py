from ultralytics import YOLO
from config import (
    ALARM_REGION,
    ALARM_REGION_ENABLED,
    ROI_IOU_THRESHOLD,
    CONF_THRESHOLD,
    FRAME_SKIP,
    MODEL_PATH,
    IMG_SIZE,
    ALARM_CLASSES
)

class AIEngine:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.conf_threshold = CONF_THRESHOLD
        self.frame_skip = FRAME_SKIP
        self.imgsz = IMG_SIZE

        self.frame_count = 0
        self.last_result = {
            "alarm": False,
            "classes": [],
            "boxes": []
        }

    def compute_iou(self, box, region):
        x1, y1, x2, y2 = box
        rx, ry, rw, rh = region.values()

        # ROI rectangle
        rx2 = rw
        ry2 = rh

        # Intersection
        inter_x1 = max(x1, rx)
        inter_y1 = max(y1, ry)
        inter_x2 = min(x2, rx2)
        inter_y2 = min(y2, ry2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        box_area = (x2 - x1) * (y2 - y1)

        if box_area == 0:
            return 0.0

        return inter_area / box_area

    def detect(self, frame):
        self.frame_count += 1

        # FRAME SKIP
        if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1) != 1):
            return self.last_result

        results = self.model(frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)

        detections = {
            "alarm": False,
            "classes": [],
            "boxes": []
        }

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                inside = False

                if ALARM_REGION_ENABLED:
                    iou = self.compute_iou((x1, y1, x2, y2), ALARM_REGION)
                    inside = iou >= ROI_IOU_THRESHOLD
                else:
                    inside = True  # ROI devredışı → tüm kutular alarm alanında

                # Alarm tetikleme koşulu
                if inside and cls_id in ALARM_CLASSES:
                    detections["alarm"] = True

                detections["classes"].append(cls_id)
                detections["boxes"].append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "inside": inside
                })

        self.last_result = detections
        return detections
