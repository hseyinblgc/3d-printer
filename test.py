from ultralytics import YOLO
import os

# 1. Modeli Yükle (best.pt dosyanızın yanına koyun veya tam yolunu yazın)
model = YOLO("best(100).pt")

# 2. Test Edilecek Klasörün Yolu
# Buraya resimlerinin olduğu klasörün tam yolunu veya adını yaz
# Örnek: "test_fotograflari" veya "C:/Kullanicilar/Adiniz/Masaustu/fotolar"
hedef_klasor = "Veri_dosyaları/hand_spaghetti/test/images" 

# Klasörün var olup olmadığını kontrol edelim
if not os.path.exists(hedef_klasor):
    print(f"HATA: '{hedef_klasor}' adında bir klasör bulunamadı!")
    print("Lütfen kodun olduğu yere bu isimde bir klasör açıp içine resim atın.")
else:
    print(f"'{hedef_klasor}' içindeki görseller işleniyor...")

    # 3. Toplu Tahmin Yap
    # stream=True: Çok fazla resim varsa hafızayı şişirmemek için kullanırız
    results = model.predict(
        source=hedef_klasor, 
        save=True,        # Çizilenleri kaydet
        conf=0.5,        # Güven eşiği
        show=False,       # Ekrana pencere açma (100 resim varsa PC donar), sadece kaydet
        name='toplu_sonuc', # Kaydedilecek klasör adı
        exist_ok=True,    # Klasör varsa üzerine yaz
        verbose= False
    )

    print(f"\n✅ İŞLEM TAMAMLANDI!")
    print(f"Sonuçları şu klasörde bulabilirsin: runs/detect/toplu_sonuc")