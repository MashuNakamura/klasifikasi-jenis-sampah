from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

# ==========================================
# INISIALISASI APP & MODEL
# ==========================================
app = FastAPI(
    title="Waste Classification API",
    description="API Endpoint untuk klasifikasi sampah Organic vs Recyclable menggunakan MobileNetV2",
    version="1.0.0"
)

# Load Model (Startup)
print("⏳ Loading Model MobileNet...")
try:
    model = load_model("waste_classifier_mobilenet.keras")
    print("✅ Model Berhasil Dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")

labels = {0: "Organic", 1: "Recyclable"}

# ==========================================
# ENDPOINT DEFINITIONS
# ==========================================

@app.get("/")
def index():
    """
    Health Check Endpoint.
    """
    return {
        "message": "Waste Classification API Ready",
        "status": "running",
        "success": True,
        "error_code": 0
    }

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Main Endpoint: Prediksi gambar dengan Error Handling Robust.
    """
    try:
        # 1. Read Image File
        contents = await file.read()

        # Mencoba membuka bytes sebagai gambar.
        # Jika file rusak/bukan gambar, baris ini akan melempar Error ke 'except'
        image = Image.open(io.BytesIO(contents))

        # 2. Preprocessing
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Inference (Prediksi)
        prediction = model.predict(img_array, verbose=0)

        # Ambil hasil
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        result_label = labels[class_idx]

        # 4. Return SUCCESS (Code 0)
        return {
            "success": True,
            "error_code": 0,
            "filename": file.filename,
            "prediction": result_label,
            "confidence": confidence,
            "message": "Prediksi berhasil dilakukan."
        }

    except Exception as e:
        # 5. Masuk sini jika file bukan gambar atau error saat processing
        return {
            "success": False,
            "error_code": 1,
            "filename": file.filename,
            "prediction": None,
            "confidence": 0.0,
            "message": f"Gagal memproses gambar. File mungkin rusak atau bukan format gambar valid. Detail: {str(e)}"
        }