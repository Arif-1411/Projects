"""
FastAPI Deployment for Multimodal AI
Accepts text and image inputs for classification
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import yaml
import uvicorn

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load model
from models.fusion_model import MultimodalFusionModel

model = MultimodalFusionModel(config)
model.load_weights("models/weights/model_best.h5")

# FastAPI app
app = FastAPI(
    title="Multimodal AI Analyzer",
    description="Text and Image Content Analysis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...)
):
    """Predict using text and image"""
    try:
        # Image processing
        image_content = await image.read()
        img = Image.open(io.BytesIO(image_content))
        img_array = np.array(img.resize(config["data"]["image_size"]))
        img_array = np.expand_dims(img_array, axis=0)
        
        # Input dictionary
        inputs = {
            'texts': tf.constant([text]),
            'images': tf.constant(img_array, dtype=tf.float32)
        }
        
        # Prediction
        predictions = model(inputs, training=False)
        probabilities = predictions.numpy()[0]
        
        # Class determination
        classes = ['Real', 'Fake']
        pred_class = classes[np.argmax(probabilities)]
        confidence = float(np.max(probabilities))
        
        return PredictionResponse(
            prediction=pred_class,
            confidence=confidence,
            class_probabilities={
                classes[0]: float(probabilities[0]),
                classes[1]: float(probabilities[1])
            }
        )
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "input_shape": {
            "text": "string",
            "image": f"{config['data']['image_size']} RGB"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)