from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import io
import os
import logging

# -------------------- CONFIGURATION -------------------- #
app = FastAPI(title="NephroScan AI Backend", version="1.0")

# CORS setup - allows frontend (Lovable) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change later to your actual frontend domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nephroscan-backend")

# Load environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/training/model.keras")
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found at path: {MODEL_PATH}")

# Load the model once at startup
logger.info("Loading model...")
model = load_model(MODEL_PATH)
logger.info("Model loaded successfully!")

# -------------------- ROUTES -------------------- #
@app.get("/")
async def root():
    """Health check route"""
    return {"message": "✅ NephroScan AI backend is running successfully!"}


@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted class & confidence score.
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Read and preprocess the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        logger.info(f"Prediction -> Class: {predicted_class}, Confidence: {confidence:.4f}")

        return {
            "status": "success",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- ENTRY POINT -------------------- #
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
