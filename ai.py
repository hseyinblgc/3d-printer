from ultralytics import YOLO

class AIEngine:
    def __init__(
        self,
        model_path="models/best.onnx",
        conf_threshold=0.4,
        frame_skip=10,     # Her 2 frame'de 1 tespit (0 = her frame)
        imgsz=640
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.frame_skip = frame_skip
        self.imgsz = imgsz

        self.frame_count = 0
        self.last_result = {
            "hand": False,
            "spaghetti": False,
            "boxes": []
        }

    def detect(self, frame):

        self.frame_count += 1

        # --- FRAME SKIP KONTROLÜ ---
        if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 1:
            return self.last_result

        results = self.model(
            frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False
        )

        detections = {
            "hand": False,
            "spaghetti": False,
            "boxes": []
        }

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id == 0:
                    detections["hand"] = True
                elif cls_id == 1:
                    detections["spaghetti"] = True
                elif cls_id == 2:
                    pass  # background → alarm yok

                detections["boxes"].append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2)
                })

        # Son sonucu hafızaya al (frame skip için)
        self.last_result = detections
        return detections
