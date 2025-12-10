import sys
import cv2
import time
import os
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, QDateTime, QUrl
from PyQt6.QtMultimedia import QSoundEffect
from _main_ui import Ui_MainWindow
from ai import AIEngine

# Ses efektini global değişkende tutuyoruz
sound_effect = None
last_detection_state = None  # Son algılanan nesne durumu

def warnuser():
    """Ses çal"""
    global sound_effect
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/media/audio.wav")
    
    if not os.path.exists(path):
        print(f"Ses dosyası bulunamadı: {path}")
        return
    
    try:
        sound_effect = QSoundEffect()
        sound_effect.setSource(QUrl.fromLocalFile(path))
        sound_effect.setVolume(1.0)
        sound_effect.play()
    except Exception as e:
        print(f"Ses çalma hatası: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # AI MOTORU
        self.ai = AIEngine()
        
        # Son algılanan nesne durumu (tespit edilmiş mi?)
        self.last_detection_state = None
        
        # Kamera başlat
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS hesaplama için değişkenler
        self.prev_frame_time = time.time()
        self.new_frame_time = 0
        self.last_fps_update_time = 0  # FPS güncelleme zamanlayıcısı
        
        # Timer ile frame güncelle
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def add_log(self, message):
        """Log mesajını textBrowser_log'a ekle"""
        timestamp = QDateTime.currentDateTime().toString('HH:mm:ss')
        log_message = f"[{timestamp}] {message}"
        self.ui.textBrowser_log.append(log_message)
    
    def update_frame(self):
        """Kameradan frame al ve pixmap'e göster"""
        ret, frame = self.camera.read()
        if not ret:
            return

        results = self.ai.detect(frame)
        
        # Mevcut algılama durumu
        current_detection = results["hand"] or results["spaghetti"]
        
        # Eğer önceki durumda algılanmamış ama şimdi algılandıysa, bir kez çal
        if not self.last_detection_state and current_detection:
            warnuser()
            
            # Tespit edilen nesneyi logla
            detected_items = []
            if results["hand"]:
                detected_items.append("EL")
            if results["spaghetti"]:
                detected_items.append("SPAGHETTI")
            
            if detected_items:
                self.add_log(f"⚠️ TESPİT EDİLDİ: {', '.join(detected_items)}")
        
        # Son durumu güncelle
        self.last_detection_state = current_detection

        for box in results["boxes"]:
            cls_id = box["class_id"]
            x1, y1, x2, y2 = box["box"]

            if cls_id == 0:
                label = "HAND"
                color = (0, 255, 0)
            elif cls_id == 1:
                label = "SPAGHETTI"
                color = (0, 0, 255)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # FPS Hesaplama
        self.new_frame_time = time.time()
        time_diff = self.new_frame_time - self.prev_frame_time
        if time_diff > 0:
            fps = 1 / time_diff
            self.prev_frame_time = self.new_frame_time

            if self.new_frame_time - self.last_fps_update_time > 0.5:
                self.ui.fps.setText(f"FPS: {int(fps)}")
                self.last_fps_update_time = self.new_frame_time

        # Tarih ve Saat Güncelleme
        current_datetime = QDateTime.currentDateTime()
        self.ui.saat.setText(current_datetime.toString('HH:mm:ss'))
        self.ui.tarih.setText(current_datetime.toString('dd.MM.yyyy'))

        # BGR'den RGB'ye dönüştür
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # OpenCV frame'i QImage'e dönüştür
        h, w, ch = rgb_frame.shape
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # QPixmap'e dönüştür ve label'a ayarla
        pixmap = QPixmap.fromImage(qt_image)
        self.ui.pixmap_media.setPixmap(pixmap.scaledToWidth(640))

    def closeEvent(self, event):
        """Pencere kapanırken kamerayı kapat"""
        self.timer.stop()
        self.camera.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())