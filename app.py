import cv2
import time
import os
from ultralytics import YOLO

ALARM_REGION = {
    "x": 364,
    "y": 385,
    "w": 957,
    "h": 586
}

ROI_RATIO = {
    "x": 364 / 1920,
    "y": 385 / 1080,
    "w": 957 / 1920,
    "h": 586 / 1080
}


ALARM_REGION_ENABLED = True
ROI_IOU_THRESHOLD = 0.5

MODEL_PATH = "resources/models/best.onnx"
CONF_THRESHOLD = 0.4
FRAME_SKIP = 2
IMG_SIZE = 640

CLASS_HAND = 0
CLASS_SPAGHETTI = 1
CLASS_BACKGROUND = 2

ALARM_CLASSES = [CLASS_HAND, CLASS_SPAGHETTI]

ALARM_SOUND_PATH = "resources/media/audio.wav"
ALARM_VOLUME = 1.0
ALARM_COOLDOWN = 1.0

CAMERA_SOURCE = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


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

    def detect(self, frame):
        self.frame_count += 1

        if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1) != 1):
            return self.last_result

        if ALARM_REGION_ENABLED:
            x, y, w, h = get_scaled_roi(frame)
            roi_frame = frame[y:y + h, x:x + w]
            results = self.model(roi_frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)
        else:
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

                if ALARM_REGION_ENABLED:
                    x1 += x
                    y1 += y
                    x2 += x
                    y2 += y

                if cls_id in ALARM_CLASSES:
                    detections["alarm"] = True

                detections["classes"].append(cls_id)
                detections["boxes"].append({
                    "class_id": cls_id,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                    "inside": True
                })

        self.last_result = detections
        return detections


import winsound
import threading

last_alarm_time = 0

def warnuser():
    global last_alarm_time

    now = time.time()
    if (now - last_alarm_time) < ALARM_COOLDOWN:
        return

    last_alarm_time = now

    if not os.path.exists(ALARM_SOUND_PATH):
        print(f"UYARI: Ses dosyası bulunamadı: {ALARM_SOUND_PATH}")
        return

    threading.Thread(
        target=winsound.PlaySound,
        args=(ALARM_SOUND_PATH, winsound.SND_FILENAME),
        daemon=True
    ).start()

def get_scaled_roi(frame):
    h, w = frame.shape[:2]

    x = int(ROI_RATIO["x"] * w)
    y = int(ROI_RATIO["y"] * h)
    rw = int(ROI_RATIO["w"] * w)
    rh = int(ROI_RATIO["h"] * h)

    return x, y, rw, rh


def main():

    ai = AIEngine()
    camera = cv2.VideoCapture(CAMERA_SOURCE)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    prev_frame_time = time.time()
    last_fps_update_time = 0

    print("3D Printer Monitor başladı...")
    print(f"Model: {MODEL_PATH}")
    print(f"Kamera: {CAMERA_SOURCE}")
    print(f"Alarm Bölgesi: {ALARM_REGION}")
    print("Çıkış için 'q' tuşuna basın\n")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Kamera okunamadı!")
            break

        results = ai.detect(frame)

        if results["alarm"]:
            warnuser()

        if ALARM_REGION_ENABLED:
            ax, ay, aw, ah = get_scaled_roi(frame)
            cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (255, 255, 0), 2)
        for box in results["boxes"]:
            cls_id = box["class_id"]
            x1, y1, x2, y2 = box["box"]

            if cls_id == CLASS_HAND:
                label = "HAND"
                color = (0, 255, 0)
            elif cls_id == CLASS_SPAGHETTI:
                label = "SPAGHETTI"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        now = time.time()
        fps = 1 / (now - prev_frame_time)
        prev_frame_time = now

        if now - last_fps_update_time > 0.5:
            detections_str = ", ".join([f"Class {cls}" for cls in results["classes"]])
            alarm_status = "ALARM!" if results["alarm"] else "Normal"
            print(f"FPS: {int(fps)} | Detections: {len(results['boxes'])} | Status: {alarm_status} | {detections_str}")
            last_fps_update_time = now

        cv2.imshow('3D Printer Monitor', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()