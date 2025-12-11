import sys
import cv2
import time
import os
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, QDateTime, QUrl
from PyQt6.QtMultimedia import QSoundEffect
from _main_ui import Ui_MainWindow

from config import (
    CAMERA_SOURCE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    DISPLAY_WIDTH,
    ALARM_REGION,
    ALARM_SOUND_PATH,
    ALARM_COOLDOWN,
    ALARM_VOLUME
)

from ai import AIEngine

sound_effect = None
last_alarm_time = 0


def warnuser():
    global sound_effect, last_alarm_time

    now = time.time()
    if (now - last_alarm_time) < ALARM_COOLDOWN:
        return  # cooldown aktif → alarm çalma

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
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # AI
        self.ai = AIEngine()

        # Camera
        self.camera = cv2.VideoCapture(CAMERA_SOURCE)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # FPS hesaplama
        self.prev_frame_time = time.time()
        self.last_fps_update_time = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)


    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            return

        # AI inference
        results = self.ai.detect(frame)

        # ALARM
        if results["alarm"]:
            warnuser()

        # ROI çizimi
        ax, ay, aw, ah = ALARM_REGION.values()
        cv2.rectangle(frame, (ax, ay), (aw, ah), (255, 255, 0), 2)

        # BBOX çizimi
        for box in results["boxes"]:
            cls_id = box["class_id"]
            inside = box["inside"]
            x1, y1, x2, y2 = box["box"]

            if cls_id == 0:
                label = "HAND"
                color = (0, 255, 0)
            elif cls_id == 1:
                label = "SPAGHETTI"
                color = (0, 0, 255)
            else:
                label = "BG"
                color = (150, 150, 150)

            if not inside:
                color = (100, 100, 100)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # FPS
        now = time.time()
        fps = 1 / (now - self.prev_frame_time)
        self.prev_frame_time = now

        if now - self.last_fps_update_time > 0.5:
            self.ui.fps.setText(f"FPS: {int(fps)}")
            self.last_fps_update_time = now

        # Timestamp
        current_datetime = QDateTime.currentDateTime()
        self.ui.saat.setText(current_datetime.toString('HH:mm:ss'))
        self.ui.tarih.setText(current_datetime.toString('dd.MM.yyyy'))

        # UI frame gösterimi
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        qt_image = QImage(rgb_frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        self.ui.pixmap_media.setPixmap(pixmap.scaledToWidth(DISPLAY_WIDTH))


    def closeEvent(self, event):
        self.camera.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
