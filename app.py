import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.set_page_config(page_title="Waste Classification", layout="wide")

# Sidebar
st.sidebar.title("♻️ Waste Classification App")
st.sidebar.markdown("""
Klasifikasikan sampah secara otomatis menggunakan CNN (MobileNet).

**Metode input:**
- Upload gambar
- Gunakan kamera langsung
""")

@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier_mobilenet.keras")

model = load_cnn_model()
labels = {0: "Organic", 1: "Recyclable"}

# Main Title
st.title("♻️ Waste Classification")
st.write("Gunakan kamera atau upload gambar sampah untuk melihat prediksi AI secara langsung.")

# =========================
# INPUT MODE
# =========================
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

image = None

with tab1:
    uploaded_file = st.file_uploader(
        "Upload image here:",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

with tab2:
    camera_image = st.camera_input("Arahkan kamera ke sampah")
    if camera_image is not None:
        image = Image.open(camera_image)

# =========================
# PREDICTION
# =========================
if image is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
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