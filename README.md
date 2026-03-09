# ⚽ VAR (Video Assistant Referee) - AI Offside Line Detector

Bu proje, bilgisayarlı görü (Computer Vision) ve makine öğrenmesi algoritmaları kullanarak futbol maçlarındaki ofsayt çizgilerini otomatik olarak çeken bir yapay zeka prototipidir. 

Gerçek dünya problemlerini çözmek amacıyla geliştirilen bu sistem; oyuncuların iskelet yapısını çıkarır, formalarına göre takımları ayırır ve saha perspektifine uygun (Homografi) dinamik bir VAR ofsayt çizgisi çeker.

## 🚀 Projenin Temel Yetenekleri

* **İskelet Çıkarma (Pose Estimation):** YOLOv8-pose modeli kullanılarak sahadaki tüm oyuncuların eklem noktaları (ayak bileği, diz, kalça vb.) milisaniyeler içinde tespit edilir.
* **Akıllı Takım Ayrımı (K-Means Clustering):** Oyuncuların formalarındaki baskın renkler analiz edilerek, `scikit-learn` K-Means algoritması ile oyuncular otomatik olarak "A Takımı" ve "B Takımı" olarak kümelenir.
* **Perspektif Dönüşümü (Homography):** Kamera açısı nedeniyle oluşan perspektif bozulmaları, `OpenCV` ile kuşbakışı (bird-eye view) düzleme aktarılır. Böylece ofsayt çizgisi havada asılı kalmaz, tam olarak saha zeminine paralel çizilir.
* **Esnek Hata Toleransı:** Oyuncunun ayak bileği görünmüyorsa algoritma diz kapağına veya kalçaya odaklanarak çizginin kaybolmasını engeller.

## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **OpenCV** (Görüntü İşleme ve Homografi matrisleri)
* **YOLOv8 by Ultralytics** (Nesne Tespiti ve İskelet Çıkarma)
* **Scikit-Learn** (K-Means Makine Öğrenmesi algoritması)
* **NumPy** (Matris ve koordinat hesaplamaları)

## 🎥 Demo
*(Buraya sistemin çalışırken kaydettiği output_var.mp4 videosundan aldığın kısa bir GIF'i veya ekran görüntüsünü ekleyebilirsin)*
<video src="output_var.mp4" width="100%" controls></video>

## 💻 Kurulum ve Çalıştırma

Projenin yerel bilgisayarında çalışması için aşağıdaki adımları izleyebilirsin:

1. Repoyu bilgisayarına klonla:
   ```bash
   git clone [https://github.com/](https://github.com/)[Senin_Kullanici_Adin]/[Repo_Adin].git
Gerekli kütüphaneleri yükle:

Bash
pip install ultralytics opencv-python scikit-learn numpy
Test videonu (test_video.mp4 adıyla) ana dizine ekle ve kodu çalıştır:

Bash
python main.py
📐 Nasıl Kullanılır? (Kalibrasyon Aşaması)
Sistem çalıştığında kamera açısını öğrenmek için video ilk karede duraklatılır:

Sahadaki referans bir dikdörtgenin (örneğin ceza sahası) 4 köşesine sırayla tıklayın (Sol Üst -> Sağ Üst -> Sağ Alt -> Sol Alt).

Tıklama bittikten sonra sistem otomatik olarak çalışmaya başlayacak ve analizi tamamladığında output_var.mp4 adında işlenmiş bir video kaydedecektir.

🚧 Bilinen Kısıtlamalar ve Gelecek Çalışmalar (Future Work)
Proje şu an statik homografi (sabit kamera açısı) ile çalışmaktadır. Yayın kamerasının dinamik olarak sağa sola döndüğü (pan/tilt) durumlarda çizginin kaymasını önlemek amacıyla, gelecekte Optical Flow veya SIFT özellik eşleştirmesi ile her karede otomatik perspektif kalibrasyonu güncelleyen bir eklenti yapılması planlanmaktadır.

Bu proje [Adil Ege Tanrıverdi] tarafından portfolyo amacıyla geliştirilmiştir.


