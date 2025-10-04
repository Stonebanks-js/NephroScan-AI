from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import io
import os
import base64
import json

app = FastAPI()

# Allow frontend (Lovable) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/training/model.keras")
model = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "NephroScan AI backend is running successfully!"}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
