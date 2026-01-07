import streamlit as st
import numpy as np
import time
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="EcoSort AI - Smart Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
/* Global Font & Background */
html, body, [class*="css"] {
    font-family: 'Segoe UI', Tahoma, sans-serif;
    background-color: #F6F7F9;
    color: #1F2937; /* dark gray */
}

/* Header */
.main-header {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1B5E20; /* dark green */
    text-align: center;
}

.sub-header {
    font-size: 1.2rem;
    text-align: center;
    color: #374151; /* readable gray */
    margin-bottom: 1.5rem;
}

/* Card Result */
.card {
    background: #FAFAFA; /* off-white */
    padding: 26px;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.card h1 {
    color: #111827; /* almost black */
    margin-bottom: 4px;
}

.card p {
    color: #4B5563; /* medium gray */
    font-size: 1.05rem;
}

/* Streamlit Info / Success / Warning */
.stAlert {
    color: #111827;
}

/* Expander text */
.streamlit-expanderContent {
    color: #1F2937;
}

/* Caption */
.stCaption {
    color: #6B7280;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=90)
    st.title("🌱 EcoSort AI")

    st.markdown("### 🎯 Tujuan Sistem")
    st.write("""
    Membantu masyarakat **mengenali jenis sampah**
    dan **cara pengolahan yang tepat** hanya dari foto.
    """)

    st.markdown("### 👥 Dapat Digunakan Oleh")
    st.write("""
    - Masyarakat umum  
    - Sekolah & Kampus  
    - Program lingkungan & Smart City  
    """)

    st.markdown("### 👨‍💻 Developer")
    st.info("""
    **Federico Matthew Pratama**  
    Computer Science  
    Universitas Katolik Darma Cendika
    """)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier_mobilenet.keras")

model = load_cnn_model()
model.predict(np.zeros((1,224,224,3)), verbose=0)

labels = {0: "Organic", 1: "Recyclable"}

# ==========================================
# HELPER
# ==========================================
def confidence_label(conf):
    if conf >= 0.85:
        return "🟢 Sangat Yakin"
    elif conf >= 0.70:
        return "🟡 Cukup Yakin"
    else:
        return "🔴 Kurang Yakin"

# ==========================================
# HEADER
# ==========================================
st.markdown('<p class="main-header">♻️ EcoSort AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI membantu mengenali jenis sampah dan memberikan saran pengolahannya</p>',
    unsafe_allow_html=True
)

# ==========================================
# MAIN LAYOUT
# ==========================================
col_input, col_result = st.columns([1,1.2], gap="large")

# ==========================================
# INPUT
# ==========================================
with col_input:
    st.subheader("📸 Ambil atau Upload Foto Sampah")

    st.info("""
    **Cara Menggunakan:** \n
    1️⃣ Ambil foto sampah  
    2️⃣ Upload atau gunakan kamera  
    3️⃣ Lihat hasil & saran pengolahan
    """)

    tab1, tab2 = st.tabs(["📁 Upload", "📷 Kamera"])
    image = None

    with tab1:
        file = st.file_uploader("Upload gambar (JPG / PNG)", type=["jpg","jpeg","png"])
        if file:
            image = Image.open(file)

    with tab2:
        cam = st.camera_input("Ambil foto")
        if cam:
            image = Image.open(cam)

    if image:
        st.image(image, caption="Gambar yang dianalisis", use_column_width=True)

# ==========================================
# RESULT
# ==========================================
with col_result:
    st.subheader("🔍 Hasil Analisis")

    if image is None:
        st.info("👈 Silakan masukkan gambar sampah terlebih dahulu")
        st.image(
            "https://cdn-icons-png.flaticon.com/512/8634/8634075.png",
            width=160
        )
    else:
        with st.spinner("AI sedang menganalisis..."):
            time.sleep(0.8)

            img = image.resize((224,224))
            arr = img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr, verbose=0)
            idx = np.argmax(pred)
            conf = pred[0][idx]

        st.markdown("## 🧠 AI Menilai Sampah Ini Sebagai:")

        if conf < 0.60:
            st.warning("🤔 Sampah Belum Bisa Dikenali")
            st.write("""
            Kemungkinan:
            - Foto kurang jelas  
            - Bukan objek sampah  
            - Sampah campuran
            """)
        else:
            if idx == 0:
                st.markdown("""
                <div class="card" style="border-left:10px solid #4CAF50;">
                    <h1>🌱 ORGANIC</h1>
                    <p>Mudah terurai secara alami</p>
                </div>
                """, unsafe_allow_html=True)

                st.success(f"Tingkat Keyakinan AI: **{confidence_label(conf)}**")

                with st.expander("✅ Saran Pengolahan", expanded=True):
                    st.write("""
                    - Buang ke komposter  
                    - Olah menjadi pupuk  
                    - Dapat digunakan untuk maggot (BSF)
                    """)
            else:
                st.markdown("""
                <div class="card" style="border-left:10px solid #2196F3;">
                    <h1>♻️ RECYCLABLE</h1>
                    <p>Dapat didaur ulang</p>
                </div>
                """, unsafe_allow_html=True)

                st.info(f"Tingkat Keyakinan AI: **{confidence_label(conf)}**")

                with st.expander("✅ Saran Pengolahan", expanded=True):
                    st.write("""
                    - Cuci & keringkan  
                    - Pisahkan tutup botol  
                    - Setorkan ke Bank Sampah
                    """)

        st.write("---")
        st.caption("Seberapa yakin AI terhadap hasil ini?")
        st.progress(float(conf))