# Klasifikasi Sampah Menggunakan CNN & MobileNet

Proyek ini adalah implementasi *Deep Learning* untuk mengklasifikasikan jenis sampah (Waste Classification) menggunakan metode **Transfer Learning** dengan arsitektur **MobileNet** sebagai *base model*. Proyek ini dibangun menggunakan Python dan TensorFlow/Keras.

## 📋 Daftar Isi
- [Deskripsi Proyek](#deskripsi-proyek)
- [Struktur Folder](#struktur-folder)
- [Prasyarat & Instalasi](#prasyarat--instalasi)
- [Dataset](#dataset)
- [Cara Penggunaan](#cara-penggunaan)
- [Hasil & Evaluasi](#hasil--evaluasi)

## 📖 Deskripsi Proyek
Sistem ini dirancang untuk mendeteksi dan mengkategorikan sampah secara otomatis dari gambar. Model dilatih menggunakan *Convolutional Neural Network* (CNN) dengan memanfaatkan *pre-trained model* MobileNet untuk efisiensi dan akurasi yang lebih baik, terutama pada perangkat dengan komputasi terbatas.

File notebook (`.ipynb`) mencakup langkah-langkah:
1. Data Preprocessing & Augmentasi.
2. Pembangunan Model (Transfer Learning MobileNet).
3. Pelatihan (Training) & Validasi.
4. Evaluasi Model.
5. Penyimpanan Model (`.h5`).

## 📂 Struktur Folder
Berikut adalah susunan direktori proyek di lingkungan lokal:

```text
Deteksi_Jenis_Sampah/
├── app.py                                   # Script Python utama untuk aplikasi Streamlit
├── KLASIFIKASI_SAMPAH_CNN_MOBILE_NET.ipynb  # Notebook Jupyter/Google Colab untuk training dan eksperimen
├── requirements.txt                         # Daftar pustaka (library) yang dibutuhkan
├── waste_classifier_mobilenet.keras         # Model CNN MobileNet yang sudah dilatih (Saved Model)
├── README.md                                # Dokumentasi singkat dan panduan proyek
```