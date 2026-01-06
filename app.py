import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import time

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Waste Classification Demo",
    page_icon="♻️",
    layout="centered"
)

# CSS Custom untuk mempercantik tampilan & RATA TENGAH
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Mengatur text-align center untuk header custom */
    .center-text {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier_mobilenet.keras")

try:
    model = load_cnn_model()
except:
    st.error("⚠️ Model tidak ditemukan! Pastikan file 'waste_classifier_mobilenet.keras' ada.")
    st.stop()

labels = {0: "Organic", 1: "Recyclable"}

# ==========================================
# HEADER (JUDUL & DESKRIPSI RATA TENGAH)
# ==========================================
# Kita pakai st.markdown dengan HTML biar bisa di-center
st.markdown("""
    <div class="center-text">
        <h1 style='color: #2e7d32;'>♻️ Waste Classification AI</h1>
        <p style='font-size: 18px;'>
            <b>Demonstrasi Project Deep Learning</b><br>
            Klasifikasi sampah Organik vs. Daur Ulang (Recyclable) menggunakan MobileNet.
        </p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# MAIN CONTENT
# ==========================================

# Tab Pilihan
st.write("") # Spasi dikit
tab_upload, tab_camera = st.tabs(["📁 Upload Gambar", "📷 Ambil Foto"])

# Fungsi Proses Gambar
def process_and_predict(image_input):
    # Tampilkan Gambar
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image_input, caption="Input Gambar", use_column_width=True)

    # Preprocessing
    img = image_input.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Efek Loading
    with col2:
        with st.spinner('🤖 AI sedang menganalisis piksel...'):
            time.sleep(1.0) # Delay dikit biar kerasa prosesnya
            prediction = model.predict(img_array, verbose=0)

        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        # --- LOGIKA SAFETY NET & TAMPILAN ---
        threshold = 0.60

        if confidence < threshold:
            st.warning("⚠️ **Objek Tidak Dikenali**")
            st.write("Fitur visual sampah tidak terdeteksi dengan jelas.")
            st.progress(float(confidence))

        else:
            label = labels[class_idx]

            # Tampilan Jika ORGANIC
            if class_idx == 0:
                st.markdown(
                    f"""
                    <div class="result-card" style="background-color: #e8f5e9; border: 2px solid #4caf50;">
                        <h2 style="color: #2e7d32;">🌱 ORGANIC</h2>
                        <p>Terdeteksi sebagai sampah organik.</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            # Tampilan Jika RECYCLABLE
            else:
                st.markdown(
                    f"""
                    <div class="result-card" style="background-color: #e3f2fd; border: 2px solid #2196f3;">
                        <h2 style="color: #1565c0;">♻️ RECYCLABLE</h2>
                        <p>Terdeteksi sebagai sampah daur ulang.</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            st.markdown(f"<p style='text-align: center; margin-top: 10px;'>Confidence: <b>{confidence:.1%}</b></p>", unsafe_allow_html=True)
            st.progress(float(confidence))

# ==========================================
# LOGIKA TAB
# ==========================================
with tab_upload:
    uploaded_file = st.file_uploader("Upload sampel gambar untuk diuji:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        process_and_predict(image)

with tab_camera:
    camera_image = st.camera_input("Arahkan kamera ke sampel sampah")
    if camera_image is not None:
        image = Image.open(camera_image)
        process_and_predict(image)