import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

st.set_page_config(page_title="Waste Classification", layout="wide")

# Sidebar
st.sidebar.title("♻️ Waste Classification App")
st.sidebar.markdown("""
Upload gambar sampah dan model akan mengklasifikasikan menjadi:
- **Organic**
- **Recyclable**
""")

@st.cache_resource
def load_cnn_model():
    return load_model("waste_classifier_mobilenet.keras")

model = load_cnn_model()
labels = {0: "Organic", 1: "Recyclable"}

# Main Title
st.title("♻️ Waste Classification")
st.write("Upload gambar sampah untuk mendapatkan prediksi jenis dan confidence.")

# Upload file
uploaded_file = st.file_uploader(
    "Upload image here:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Layout 2 kolom
    col1, col2 = st.columns([1, 1])

    # Tampilkan gambar
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediksi
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        prediction = model.predict(img_array, verbose=0)

    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    # Hasil prediksi
    with col2:
        st.subheader("Prediction Result")
        st.metric(label="Class", value=labels[class_idx])
        st.metric(label="Confidence", value=f"{confidence:.2f}")