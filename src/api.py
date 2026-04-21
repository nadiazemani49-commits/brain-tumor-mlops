import os
import torch
import timm
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io

CLASSES       = ['glioma', 'meningioma', 'pituitary', 'notumor']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

app   = FastAPI(title='Brain Tumor Classifier API', version='1.0.0')
MODEL = None

@app.on_event('startup')
def load_model():
    global MODEL
    model_path = Path('best_model.pth')
    MODEL = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    if model_path.exists():
        MODEL.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Modele charge depuis best_model.pth')
    MODEL.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class PredictionResult(BaseModel):
    predicted_class : str
    confidence      : float
    probabilities   : dict

@app.get('/')
def root():
    return {'message': 'Brain Tumor Classifier API', 'status': 'running'}

@app.post('/predict', response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img       = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor    = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(MODEL(tensor), dim=1).squeeze().numpy()
    pred_idx = int(probs.argmax())
    return PredictionResult(
        predicted_class = CLASSES[pred_idx],
        confidence      = float(probs[pred_idx]),
        probabilities   = {c: float(p) for c, p in zip(CLASSES, probs)},
    )

@app.get('/health')
def health():
    return {'status': 'healthy', 'model_loaded': MODEL is not None}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
