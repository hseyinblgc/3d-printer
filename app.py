import sys
import cv2
import time
import os
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, QDateTime, QUrl
from PyQt6.QtMultimedia import QSoundEffect
from _main_ui import Ui_MainWindow

# Ses efektini global değişkende tutuyoruz
sound_effect = None

def cleanup_sound():
    """Ses çalma bittiğinde nesneyi hafızadan temizle"""
    global sound_effect
    if sound_effect and not sound_effect.isPlaying():
        sound_effect = None

def warnuser():
    global sound_effect
    # Dosya yolunu tam olarak belirtmek daha güvenlidir
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/audio.wav")
    
    sound_effect = QSoundEffect()
    sound_effect.setSource(QUrl.fromLocalFile(path))
    sound_effect.setVolume(1.0)
    
    # Çalma durumu değiştiğinde (bittiğinde) temizlik fonksiyonunu tetikle
    sound_effect.playingChanged.connect(cleanup_sound)
    
    sound_effect.play()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Test etmek isterseniz warnuser'ı burada çağırabilirsiniz
        # warnuser()
        
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
        self.timer.start(30)  # 30ms = ~33 FPS
    
    def update_frame(self):
        """Kameradan frame al ve pixmap'e göster"""
        ret, frame = self.camera.read()
        
        if ret:
            # FPS Hesaplama
            self.new_frame_time = time.time()
            time_diff = self.new_frame_time - self.prev_frame_time
            if time_diff > 0:
                fps = 1 / time_diff
                self.prev_frame_time = self.new_frame_time
                
                # FPS göstergesini her 0.5 saniyede bir güncelle
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