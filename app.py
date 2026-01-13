import streamlit as st
import numpy as np
import time
import os
import warnings
from PIL import Image

# ==========================================
# 0. KONFIGURASI AWAL (ANTI WARNING)
# ==========================================
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. CUSTOM CSS (DARK MODE 2026)
# ==========================================
st.markdown("""
<style>
/* Background & Text Global */
.stApp {
    background-color: #0E1117;
    color: #FAFAFA;
}

/* Header Gradient */
.main-header {
    font-size: 2.5rem;
    font-weight: 800;
    background: -webkit-linear-gradient(#4CAF50, #81C784);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
}

.sub-header {
    font-size: 1.1rem;
    text-align: center;
    color: #B0B0B0;
    margin-bottom: 30px;
}

/* Card Edukasi */
.info-card {
    background-color: #262730;
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #363945;
    margin-bottom: 15px;
    min-height: 220px; /* Tinggi Minimal Sama */
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.info-card h4 {
    color: #FFFFFF !important;
    font-weight: 700;
    margin-bottom: 15px;
}

.info-card p, .info-card li {
    color: #E0E0E0 !important;
    font-size: 1rem;
    line-height: 1.6;
}

/* Result Card */
.result-card {
    background-color: #1F1F1F;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.result-title {
    font-size: 2.2rem;
    font-weight: 900;
    margin: 10px 0;
}

/* Tabs Button */
button[data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier_mobilenet.keras")

try:
    model = load_cnn_model()
    model.predict(np.zeros((1,224,224,3)), verbose=0)
except:
    pass

# ==========================================
# 4. HELPER
# ==========================================
def confidence_label(conf):
    if conf >= 0.85: return "🟢 Sangat Yakin"
    elif conf >= 0.70: return "🟡 Cukup Yakin"
    else: return "🔴 Kurang Yakin"

# ==========================================
# 5. HEADER
# ==========================================
st.markdown('<p class="main-header">♻️ EcoSort AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Solusi Pintar Klasifikasi Sampah Masa Depan</p>', unsafe_allow_html=True)

# ==========================================
# 6. NAVIGASI UTAMA
# ==========================================
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📸 Scan AI", "ℹ️ Tentang"])

# ==========================================
# TAB 1: BERANDA (EDUKASI)
# ==========================================
with tab1:
    st.write("### 🌏 Wawasan Lingkungan")

    # --- ROW 1: TEXT CARDS (SAMA TINGGI) ---
    col_text1, col_text2 = st.columns(2)

    with col_text1:
        st.markdown("""
        <div class="info-card" style="border-left: 5px solid #FFEB3B;">
            <h4>⚠️ Masalah: Tumpukan Sampah</h4>
            <p>Sampah yang tercampur di TPA menghasilkan gas metana berbahaya. Jutaan ton sampah berakhir mencemari laut dan tanah kita setiap tahunnya.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_text2:
        st.markdown("""
        <div class="info-card" style="border-left: 5px solid #4CAF50;">
            <h4>✅ Solusi: Teknologi AI</h4>
            <p>EcoSort hadir menggunakan teknologi <i>Computer Vision</i> untuk memilah sampah secara instan, membantu proses daur ulang menjadi lebih efisien.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW 2: IMAGES (PRESISI 800x500) ---
    st.divider()
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        # Gambar Masalah (Auto Crop 800x500)
        st.image("https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?auto=format&fit=crop&w=800&h=500&q=80",
                 caption="Realita: Sampah Tercampur",
                 use_container_width=True)

    with col_img2:
        # Gambar Solusi (Auto Crop 800x500)
        st.image("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?auto=format&fit=crop&w=800&h=500&q=80",
                 caption="Harapan: Lingkungan Bersih",
                 use_container_width=True)

    st.divider()
    st.write("### 📚 Kategori Deteksi")

    # --- ROW 3: LIST CARDS (SAMA TINGGI) ---
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card" style="border: 1px solid #4CAF50;">
            <h4 style="color:#66BB6A !important;">🍂 ORGANIK</h4>
            <ul>
                <li>Sisa Makanan & Tulang</li>
                <li>Kulit Buah & Sayur</li>
                <li>Daun & Ranting Pohon</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card" style="border: 1px solid #2196F3;">
            <h4 style="color:#42A5F5 !important;">♻️ RECYCLABLE</h4>
            <ul>
                <li>Botol Plastik & Kaca</li>
                <li>Kertas, Kardus, Koran</li>
                <li>Kaleng Minuman & Logam</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# TAB 2: SCAN AI
# ==========================================
with tab2:
    st.write("### 🤖 Deteksi Sampah")
    st.caption("Pilih metode pengambilan gambar:")

    with st.container(border=True):
        input_type = st.radio("", ["📁 Upload File", "📷 Kamera"], horizontal=True, label_visibility="collapsed")

        image = None
        if input_type == "📁 Upload File":
            file = st.file_uploader("Upload foto (JPG/PNG)", type=["jpg","jpeg","png"])
            if file: image = Image.open(file)
        else:
            cam = st.camera_input("Jepret foto")
            if cam: image = Image.open(cam)

    if image:
        st.divider()
        st.write("##### 🔍 Hasil Analisis")

        col_img, col_info = st.columns([1, 1.5])

        with col_img:
            st.image(image, caption="Input Citra", use_container_width=True)

        with col_info:
            with st.spinner("Sedang memproses..."):
                time.sleep(0.5)
                img_resized = image.resize((224,224))
                arr = img_to_array(img_resized) / 255.0
                arr = np.expand_dims(arr, axis=0)

                pred = model.predict(arr, verbose=0)
                idx = np.argmax(pred)
                conf = pred[0][idx]

            if conf < 0.60:
                st.warning("⚠️ Objek tidak dikenal. Coba foto lebih jelas.")
            else:
                label = "ORGANIC" if idx == 0 else "RECYCLABLE"
                color = "#66BB6A" if idx == 0 else "#42A5F5"

                st.markdown(f"""
                <div class="result-card" style="border: 2px solid {color};">
                    <p style="color:#BBB; margin:0;">Terdeteksi Sebagai</p>
                    <h1 class="result-title" style="color: {color};">{label}</h1>
                    <p style="color:#EEE; font-weight:bold;">{confidence_label(conf)} ({conf*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                if idx == 0:
                    with st.expander("🌱 Lihat Saran Pengolahan"):
                        st.info("Bisa dijadikan **Pupuk Kompos** atau pakan Maggot (BSF).")
                else:
                    with st.expander("♻️ Lihat Saran Pengolahan"):
                        st.info("Cuci bersih, pilah tutup botol, dan bawa ke **Bank Sampah**.")

    else:
        st.info("👆 Masukkan gambar untuk memulai.")

# ==========================================
# TAB 3: TENTANG
# ==========================================
with tab3:
    st.write("### 👨‍💻 Developer Profile")

    col_a, col_b = st.columns([1, 4])
    with col_a:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png")
    with col_b:
        st.markdown("""
        **Federico Matthew Pratama** Computer Science Student  
        *Universitas Katolik Darma Cendika*
        """)

        st.link_button(
            "Buka Repository GitHub",
            "https://github.com/MashuNakamura/klasifikasi-jenis-sampah",
            type="primary"
        )

    st.divider()
    st.write("### ℹ️ Tentang Aplikasi")
    st.write("""
    **EcoSort AI** adalah prototipe sistem pemilahan sampah cerdas berbasis *Deep Learning*.
    Dikembangkan untuk mendukung program *Smart City* dan manajemen limbah berkelanjutan.
    
    **Teknologi:**
    - Python & Streamlit (Frontend)
    - TensorFlow & MobileNetV2 (AI Core)
    - FastAPI (Backend Service)
    """)

    st.caption("© 2026 EcoSort AI. All Rights Reserved.")