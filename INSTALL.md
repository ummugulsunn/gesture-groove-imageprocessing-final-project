# GestureGroove Installation Guide
# GestureGroove Kurulum Rehberi

## English

### Prerequisites
- Python 3.8-3.11 (Python 3.13 is not supported yet)
- pip (Python package manager)
- Git (for cloning the repository)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/gesturegroove.git
   cd gesturegroove
   ```

2. **Create Virtual Environment**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MediaPipe (Important!)**
   ```bash
   # For Windows/Linux
   pip install mediapipe

   # For Apple Silicon (M1/M2/M3) Macs
   pip install mediapipe-silicon
   ```

5. **Run the Application**
   ```bash
   python src/main.py
   ```

### Troubleshooting

1. **MediaPipe Installation Issues**
   - If you're using Python 3.13, downgrade to Python 3.11
   - For Apple Silicon Macs, use `mediapipe-silicon` package
   - If installation fails, try: `pip install --upgrade pip`

2. **Camera Access**
   - Ensure your camera is not being used by another application
   - Grant camera permissions to Python/your terminal

3. **Performance Issues**
   - Close other applications using the camera
   - Ensure good lighting conditions
   - Keep your hand within the camera frame

### System Requirements

- **Minimum:**
  - CPU: Dual-core 2.0 GHz
  - RAM: 4GB
  - Camera: 720p
  - OS: Windows 10, macOS 10.15+, Ubuntu 20.04+

- **Recommended:**
  - CPU: Quad-core 2.5 GHz+
  - RAM: 8GB+
  - Camera: 1080p
  - GPU: NVIDIA/AMD dedicated graphics (for better performance)

---

## Türkçe

### Ön Gereksinimler
- Python 3.8-3.11 (Python 3.13 henüz desteklenmiyor)
- pip (Python paket yöneticisi)
- Git (depoyu klonlamak için)

### Kurulum Adımları

1. **Depoyu Klonlama**
   ```bash
   git clone https://github.com/yourusername/gesturegroove.git
   cd gesturegroove
   ```

2. **Sanal Ortam Oluşturma**
   ```bash
   # Windows için
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux için
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Bağımlılıkları Yükleme**
   ```bash
   pip install -r requirements.txt
   ```

4. **MediaPipe Kurulumu (Önemli!)**
   ```bash
   # Windows/Linux için
   pip install mediapipe

   # Apple Silicon (M1/M2/M3) Mac'ler için
   pip install mediapipe-silicon
   ```

5. **Uygulamayı Çalıştırma**
   ```bash
   python src/main.py
   ```

### Sorun Giderme

1. **MediaPipe Kurulum Sorunları**
   - Python 3.13 kullanıyorsanız, Python 3.11'e geçiş yapın
   - Apple Silicon Mac'ler için `mediapipe-silicon` paketini kullanın
   - Kurulum başarısız olursa: `pip install --upgrade pip` deneyin

2. **Kamera Erişimi**
   - Kameranın başka bir uygulama tarafından kullanılmadığından emin olun
   - Python/terminal uygulamanıza kamera izinlerini verin

3. **Performans Sorunları**
   - Kamerayı kullanan diğer uygulamaları kapatın
   - İyi aydınlatma koşulları sağlayın
   - Elinizi kamera çerçevesi içinde tutun

### Sistem Gereksinimleri

- **Minimum:**
  - İşlemci: Çift çekirdekli 2.0 GHz
  - RAM: 4GB
  - Kamera: 720p
  - İşletim Sistemi: Windows 10, macOS 10.15+, Ubuntu 20.04+

- **Önerilen:**
  - İşlemci: Dört çekirdekli 2.5 GHz+
  - RAM: 8GB+
  - Kamera: 1080p
  - GPU: NVIDIA/AMD özel grafik kartı (daha iyi performans için) 