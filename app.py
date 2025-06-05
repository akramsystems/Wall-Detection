#!/usr/bin/env python3
import os
import base64
import io
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from scipy.ndimage import maximum_filter
import logging
import uvicorn
from typing import Optional, List, Dict, Any
import asyncio
import aiohttp
import tempfile

# Cloud storage imports
from azure.storage.blob import BlobServiceClient
from google.cloud import storage as gcs

# Add current directory to path
import sys
sys.path.append(os.getcwd())
from floortrans.models import get_model
from floortrans.post_prosessing import split_prediction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wall Detection API",
    description="API for detecting wall corners and joints in floor plan images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = None

class ImageRequest(BaseModel):
    image_data: Optional[str] = None  # base64 encoded image
    cloud_path: Optional[str] = None  # path to image in cloud storage
    storage_type: Optional[str] = "azure"  # "azure" or "gcp"
    threshold: Optional[float] = 0.1
    nms_size: Optional[int] = 3

class WallCorner(BaseModel):
    x: float
    y: float
    confidence: float
    corner_type: int

class InferenceResponse(BaseModel):
    wall_corners: List[WallCorner]
    image_size: Dict[str, int]
    processing_time: float

def load_model():
    """Load the pre-trained model."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = get_model('hg_furukawa_original', 51)
    
    n_classes = 44
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    
    # Load model weights
    model_path = './pkl_file/model_best_val_loss_var.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")

async def download_from_azure(blob_path: str) -> bytes:
    """Download image from Azure Blob Storage."""
    # Azure Blob Storage configuration
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "images")
    
    if not account_name or not account_key:
        raise HTTPException(status_code=400, detail="Azure storage credentials not configured")
    
    try:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_path
        )
        blob_data = blob_client.download_blob().readall()
        return blob_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from Azure: {str(e)}")

async def download_from_gcp(blob_path: str) -> bytes:
    """Download image from Google Cloud Storage."""
    # GCP Storage configuration
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not bucket_name:
        raise HTTPException(status_code=400, detail="GCP bucket name not configured")
    
    try:
        if credentials_path and os.path.exists(credentials_path):
            client = gcs.Client.from_service_account_json(credentials_path)
        else:
            client = gcs.Client()  # Use default credentials
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob_data = blob.download_as_bytes()
        return blob_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download from GCP: {str(e)}")

def extract_wall_corners(heatmaps: np.ndarray, threshold: float = 0.1, nms_size: int = 3) -> List[WallCorner]:
    """Extract wall corner coordinates from heatmaps."""
    wall_corners = []
    for c in range(13):  # channels 0-12 for wall junctions
        channel = heatmaps[c]
        max_filt = maximum_filter(channel, size=nms_size)
        peaks = (channel == max_filt) & (channel > threshold)
        ys, xs = np.where(peaks)
        for x, y in zip(xs, ys):
            confidence = float(channel[y, x])
            wall_corners.append(WallCorner(
                x=float(x),
                y=float(y),
                confidence=confidence,
                corner_type=int(c)
            ))
    return wall_corners

async def process_image(image_data: bytes, threshold: float, nms_size: int) -> InferenceResponse:
    """Process image and return wall corners."""
    import time
    start_time = time.time()
    
    # Load and preprocess image
    img = Image.open(io.BytesIO(image_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    preprocess = transforms.Compose([transforms.ToTensor()])
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        pred = model(image_tensor)
    
    # Extract heatmaps
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    img_size = (height, width)
    split = [21, 12, 11]
    heatmaps, _, _ = split_prediction(pred, img_size, split)
    
    # Extract wall corners/joints
    wall_corners = extract_wall_corners(heatmaps, threshold, nms_size)
    
    processing_time = time.time() - start_time
    
    return InferenceResponse(
        wall_corners=wall_corners,
        image_size={"width": img.width, "height": img.height},
        processing_time=processing_time
    )

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Wall Detection API is running", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict_wall_corners(request: ImageRequest):
    """
    Predict wall corners from an image.
    
    Supports two input methods:
    1. Base64 encoded image in 'image_data' field
    2. Cloud storage path in 'cloud_path' field with 'storage_type' ("azure" or "gcp")
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    image_data = None
    
    # Handle base64 image data
    if request.image_data:
        try:
            image_data = base64.b64decode(request.image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
    
    # Handle cloud storage path
    elif request.cloud_path:
        if request.storage_type == "azure":
            image_data = await download_from_azure(request.cloud_path)
        elif request.storage_type == "gcp":
            image_data = await download_from_gcp(request.cloud_path)
        else:
            raise HTTPException(status_code=400, detail="storage_type must be 'azure' or 'gcp'")
    
    else:
        raise HTTPException(status_code=400, detail="Either image_data or cloud_path must be provided")
    
    # Process the image
    try:
        result = await process_image(image_data, request.threshold, request.nms_size)
        return result
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 