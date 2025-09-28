#  Iris MLOps Pipeline  

Bu proje, **Iris veri seti** üzerinde uçtan uca bir **MLOps pipeline** oluşturmayı amaçlamaktadır.  
Amaç, veri indirip işlemekten başlayarak model eğitimi, deney takibi ve versiyonlama süreçlerini profesyonel şekilde göstermektir.  

---

## 🚀 Özellikler
- 📥 **Veri İndirme:** `src/download_data.py` ile Iris veri seti otomatik indirilir.  
- 🧹 **Veri Temizleme & Feature Engineering:** `src/clean_data.py` eksik değerleri kontrol eder, yeni özellikler üretir.  
- 🤖 **Model Eğitimi:** Logistic Regression ve Random Forest modelleri eğitilir.  
- 📊 **Deney Takibi:** MLflow ile deney sonuçları (accuracy, parametreler, metrikler) kaydedilir.  
- 📂 **Versiyonlama:** DVC ile veriler ve modeller izlenir.  

---

## 📁 Proje Yapısı
```bash
iris-mlops/
├── data/               # Ham ve işlenmiş veriler
│   ├── raw/            # Orijinal veri
│   └── processed/      # Temizlenmiş veri
├── models/             # Eğitimli modeller ve metrikler
├── notebooks/          # Veri işleme scriptleri
├── src/                # Asıl proje kaynak kodları
│   ├── download_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   └── train_model_mlflow.py
├── tests/              # Testler
│   └── test_model.py
├── dvc.yaml            # DVC pipeline tanımı
├── dvc.lock            # DVC lock dosyası
├── requirements.txt    # Gerekli kütüphaneler
└── README.md           # Proje dokümantasyonu
```

# ⚙️ Kurulum

Öncelikle gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```
## 🔄 DVC Pipeline Çalıştır

Veri hazırlama ve model eğitim pipeline’ını DVC ile çalıştırabilirsiniz:

```bash
# Veri hazırlama + model eğitimi
dvc repro
```
# Sonuçları kontrol et
```bash
dvc metrics show
```
# 🧪 Kullanım

## 1️⃣ Veri İndirme
```bash
python src/download_data.py

## 2️⃣ 🧹 Veri Temizleme

Veri temizleme adımında, ham veriler analiz ve modelleme için hazır hâle getirilir.  

Çalıştırmak için:

```bash
python src/clean_data.py
```
## 3️⃣ 🏋️ Model Eğitimi

Model eğitimi adımı, veriyi kullanarak makine öğrenmesi modellerinin oluşturulmasını ve değerlendirilmesini içerir.  

Çalıştırmak için:

```bash
python src/train_model.py
```
## 4️⃣ 📊 MLflow ile Deney Takibi

MLflow ile model deneylerinizi kaydedebilir ve takip edebilirsiniz.  

Çalıştırmak için:

```bash
python src/train_model_mlflow.py
mlflow ui
```
Ardından tarayıcıda MLflow arayüzünü açın:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

MLflow ile yapılan işlemler:  

- Deneylerin kaydedilmesi  
- Model parametrelerinin ve metriklerinin izlenmesi  
- Farklı modellerin performans karşılaştırmaları  
- Modelin kaydedilmesi ve versiyonlanması

# 🚀 Gelecek Geliştirmeler

- FastAPI ile REST API entegrasyonu  
- Docker ile konteynerleştirme  
- GitHub Actions ile CI/CD süreci

