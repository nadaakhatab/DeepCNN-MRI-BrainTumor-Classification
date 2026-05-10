import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import io
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroScan - Brain Tumor Detection")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (your front.html)
app.mount("/static", StaticFiles(directory="."), name="static")

class DeepCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    with open("classes.json", "r") as f:
        classes = json.load(f)
    logger.info(f"Loaded classes: {classes}")
except:
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

model = DeepCNN(num_classes=len(classes))

# Load model (supports both names)
for model_file in ["model.pth", "best_model.pth"]:
    if Path(model_file).exists():
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            logger.info(f"✅ Model loaded successfully from {model_file}")
            break
        except Exception as e:
            logger.error(f"Error loading {model_file}: {e}")
else:
    raise RuntimeError("Model file not found!")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def predict_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    return {
        "success": True,
        "prediction": classes[idx],
        "confidence": float(probs[idx]),
        "all_probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))}
    }

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("front.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="File must be an image")
    
    image_bytes = await file.read()
    return predict_image(image_bytes)

@app.get("/health")
async def health():
    return {"status": "healthy"}
