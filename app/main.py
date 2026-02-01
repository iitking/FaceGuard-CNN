import io
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from PIL import Image

from app.config import (
    APP_NAME,
    APP_TITLE,
    APP_DESCRIPTION,
    VERSION,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_UPLOAD_SIZE_MB,
    MODEL_METRICS,
    LABEL_MAP
)
from app.model import FaceMaskClassifier
from app.detector import (
    process_and_predict,
    parse_base64_image
)
from app.schemas import (
    PredictionResponse,
    Base64ImageRequest,
    HealthResponse,
    ModelInfoResponse
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("FaceGuard.API")

# Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FaceGuard-CNN server...")
    # Preload the model into memory
    classifier = FaceMaskClassifier.get_instance()
    if classifier.is_loaded:
        logger.info("FaceGuard-CNN model loaded successfully into memory.")
    else:
        logger.warning(
            "FaceGuard-CNN model could not be pre-loaded at startup. "
            "Ensure 'FaceGuard-CNN_model.keras' exists and TensorFlow/Keras is installed."
        )
    yield
    logger.info("Shutting down FaceGuard-CNN server...")

# Initialize FastAPI application
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for external frontend or mobile integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for static assets and templates
BASE_APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_APP_DIR / "static"
TEMPLATES_DIR = BASE_APP_DIR / "templates"

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, summary="Serve Web Dashboard")
async def serve_dashboard():
    """Serves the interactive FaceGuard-CNN web application interface."""
    index_file = TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dashboard template index.html not found."
        )
    with open(index_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Returns the operational status of the service and model readiness."""
    classifier = FaceMaskClassifier.get_instance()
    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        model_loaded=classifier.is_loaded,
        model_path=str(classifier.model_path),
        version=VERSION,
        architecture="Custom 4-Block CNN with Data Augmentation & BatchNorm"
    )


@app.get("/api/info", response_model=ModelInfoResponse, summary="Model Metadata & Performance")
async def get_model_info():
    """Returns model architecture details, input resolution, and test benchmarks."""
    return ModelInfoResponse(
        app_name=APP_NAME,
        version=VERSION,
        classes=LABEL_MAP,
        input_resolution="128x128x3 (RGB)",
        metrics=MODEL_METRICS
    )


@app.post("/api/predict", response_model=PredictionResponse, summary="Predict from Image File")
async def predict_file(
    file: UploadFile = File(..., description="Image file (.jpg, .jpeg, .png, .webp)"),
    detect_faces: bool = Query(True, description="Enable automatic face detection and cropping")
):
    """Upload an image file (single or multi-person photo) to detect face mask compliance."""
    classifier = FaceMaskClassifier.get_instance()
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure model file and dependencies are available."
        )

    # Validate file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        )

    # Read and validate image content
    try:
        contents = await file.read()
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_MB}MB."
            )

        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as e:
        logger.error(f"Failed to decode uploaded image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file or corrupted image data."
        )

    # Run detection and classification
    try:
        result = process_and_predict(image, classifier, enable_face_crop=detect_faces)
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction: {str(e)}"
        )


@app.post("/api/predict-base64", response_model=PredictionResponse, summary="Predict from Base64 Image (Webcam)")
async def predict_base64(
    request: Base64ImageRequest,
    detect_faces: bool = Query(True, description="Enable automatic face detection and cropping")
):
    """Processes a Base64-encoded image frame. Ideal for real-time webcam streaming."""
    classifier = FaceMaskClassifier.get_instance()
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    try:
        image = parse_base64_image(request.image)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {str(e)}"
        )

    try:
        result = process_and_predict(image, classifier, enable_face_crop=detect_faces)
        return result
    except Exception as e:
        logger.error(f"Base64 inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing frame: {str(e)}"
        )
