import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.set_page_config(page_title="Waste Classification", layout="wide")

# Sidebar
st.sidebar.title("♻️ Waste Classification App")
st.sidebar.markdown("""
Klasifikasikan sampah menggunakan CNN (MobileNet).

Pilih salah satu metode input:
- Upload Image
- Camera
""")

@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier_mobilenet.keras")

model = load_cnn_model()
labels = {0: "Organic", 1: "Recyclable"}

st.title("♻️ Waste Classification")
st.write("Sistem klasifikasi sampah berbasis AI dengan input gambar atau kamera.")

# =========================
# TABS
# =========================
tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera"])

# =========================
# UPLOAD MODE
# =========================
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload image:",
        type=["jpg", "jpeg", "png"],
        key="upload_input"
    )

    if uploaded_file is not None:
        image_upload = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image_upload, caption="Uploaded Image", use_column_width=True)

        img = image_upload.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("AI is analyzing..."):
            prediction = model.predict(img_array, verbose=0)

        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        with col2:
            st.subheader("Prediction Result")
            st.metric("Class", labels[class_idx])
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.2f}")

# =========================
# CAMERA MODE
# =========================
with tab_camera:
    camera_image = st.camera_input(
        "Arahkan kamera ke sampah",
        key="camera_input"
    )

    if camera_image is not None:
        image_camera = Image.open(camera_image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image_camera, caption="Camera Capture", use_column_width=True)

        img = image_camera.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("AI is analyzing..."):
            prediction = model.predict(img_array, verbose=0)

        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx]

        with col2:
            st.subheader("Prediction Result")
            st.metric("Class", labels[class_idx])
            st.progress(float(confidence))
            st.write(f"**Confidence:** {confidence:.2f}")