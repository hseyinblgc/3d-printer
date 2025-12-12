import cv2
from ultralytics import YOLO
import time

# 1. ONNX Modelini Yükle
# Dosya adının tam olarak doğru olduğundan ve kodla aynı klasörde olduğundan emin ol.
print("Model yükleniyor...")
model = YOLO('best(100).onnx', task='detect') 
print("Model yüklendi! Kamera açılıyor...")

# 2. Kamerayı Başlat (0 genelde varsayılan webcam'dir)
cap = cv2.VideoCapture(0)

# Kamera çözünürlüğünü ayarlayalım (İsteğe bağlı)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS hesaplaması için zamanlayıcı
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    # 3. TAHMİN (INFERENCE) YAP
    # imgsz=640 olarak export etmiştik, burada da belirtiyoruz.
    # conf=0.5 -> %50'nin altındaki ihtimalleri gösterme demek.
    results = model(frame, imgsz=640, conf=0.5, verbose=False)

    # 4. Sonuçları Görüntünün Üzerine Çiz
    # Ultralytics bunu bizim için otomatik yapar
    annotated_frame = results[0].plot()

    # --- FPS Hesaplama ve Ekrana Yazma ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # -------------------------------------

    # 5. Görüntüyü Ekrana Yansıt
    cv2.imshow("ONNX Model Testi - Çıkış için 'q' bas", annotated_frame)

    # 'q' tuşuna basılınca döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()