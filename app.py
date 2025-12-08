import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from _main_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Kamera başlat
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Timer ile frame güncelle
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30ms = ~33 FPS
    
    def update_frame(self):
        """Kameradan frame al ve pixmap'e göster"""
        ret, frame = self.camera.read()
        
        if ret:
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