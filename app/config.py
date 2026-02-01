import os
from pathlib import Path
from typing import Dict

# Base Directory of Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Application Information
APP_NAME = "FaceGuard-CNN"
APP_TITLE = "FaceGuard CNN | AI Face Mask Detection System"
APP_DESCRIPTION = (
    "Production-ready deep learning system for real-time face mask detection "
    "powered by a custom Convolutional Neural Network (CNN) and FastAPI."
)
VERSION = "1.0.0"

# Server Settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

# Model Configuration
MODEL_FILENAME = "FaceGuard-CNN_model.keras"
FALLBACK_MODEL_FILENAME = "FaceGuard-CNN_model.pkl"

DEFAULT_MODEL_PATH = BASE_DIR / MODEL_FILENAME
FALLBACK_MODEL_PATH = BASE_DIR / FALLBACK_MODEL_FILENAME

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

# Image Preprocessing Settings
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Class Labels mapping as trained in the model
# 0 -> Without Mask, 1 -> With Mask
LABEL_MAP: Dict[int, str] = {
    0: "Without Mask",
    1: "With Mask"
}

# Detection & Classification Thresholds
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 15))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Performance Metrics from Notebook Evaluation
MODEL_METRICS = {
    "test_accuracy": 0.9828,
    "test_loss": 0.0548,
    "epochs_trained": 30,
    "dataset": "Kaggle Face Mask Dataset (7,553 images)",
    "input_resolution": "128x128x3",
    "architecture": "4-Block Conv2D + BatchNorm + MaxPool + Dense(128) + Dense(64) + Softmax(2)"
}
