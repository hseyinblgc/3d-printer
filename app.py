import sys
import cv2
import time
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer, QDateTime, QUrl, Qt
from PyQt6.QtMultimedia import QSoundEffect
from ultralytics import YOLO

ALARM_REGION = {
    "x": 364,
    "y": 385,
    "w": 957,
    "h": 586
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

ALARM_SOUND_PATH = "resources/audio.wav"
ALARM_VOLUME = 1.0
ALARM_COOLDOWN = 1.0

CAMERA_SOURCE = 0
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

DISPLAY_WIDTH = 640


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

        rx2 = rw
        ry2 = rh

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

        if self.frame_skip > 0 and (self.frame_count % (self.frame_skip + 1) != 1):
            return self.last_result

        if ALARM_REGION_ENABLED:
            x, y, w, h = ALARM_REGION.values()
            roi_frame = frame[y:h, x:w]
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
                    x1 += ALARM_REGION["x"]
                    y1 += ALARM_REGION["y"]
                    x2 += ALARM_REGION["x"]
                    y2 += ALARM_REGION["y"]


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


sound_effect = None
last_alarm_time = 0


def warnuser():
    print("allarm")
    global sound_effect, last_alarm_time

    now = time.time()
    if (now - last_alarm_time) < ALARM_COOLDOWN:
        return

    last_alarm_time = now

    if sound_effect and sound_effect.isPlaying():
        return

    sound_effect = QSoundEffect()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ALARM_SOUND_PATH)
    sound_effect.setSource(QUrl.fromLocalFile(path))
    sound_effect.setVolume(ALARM_VOLUME)
    sound_effect.play()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Printer Monitor")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setFont(QFont("Arial", 12))
        layout.addWidget(self.fps_label)

        self.date_label = QLabel()
        self.date_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.date_label)

        self.time_label = QLabel()
        self.time_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.time_label)

        self.ai = AIEngine()

        self.camera = cv2.VideoCapture(CAMERA_SOURCE)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.prev_frame_time = time.time()
        self.last_fps_update_time = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        results = self.ai.detect(frame)

        if results["alarm"]:
            warnuser()

        if ALARM_REGION_ENABLED:
            ax, ay, aw, ah = ALARM_REGION.values()
            cv2.rectangle(frame, (ax, ay), (aw, ah), (255, 255, 0), 2)

        for box in results["boxes"]:
            cls_id = box["class_id"]
            x1, y1, x2, y2 = box["box"]

            if cls_id == CLASS_HAND:
                label = "HAND"
                color = (0, 255, 0)
            elif cls_id == CLASS_SPAGHETTI:
                label = "SPAGHETTI"
                color = (0, 0, 255)
            else:
                label = "BG"
                color = (150, 150, 150)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        now = time.time()
        fps = 1 / (now - self.prev_frame_time)
        self.prev_frame_time = now

        if now - self.last_fps_update_time > 0.5:
            self.fps_label.setText(f"FPS: {int(fps)}")
            self.last_fps_update_time = now

        current_datetime = QDateTime.currentDateTime()
        self.time_label.setText(current_datetime.toString('HH:mm:ss'))
        self.date_label.setText(current_datetime.toString('dd.MM.yyyy'))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaledToWidth(DISPLAY_WIDTH))

    def closeEvent(self, event):
        self.camera.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())