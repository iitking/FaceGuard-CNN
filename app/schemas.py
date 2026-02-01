from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    width: int = Field(..., description="Width of bounding box")
    height: int = Field(..., description="Height of bounding box")

class FacePrediction(BaseModel):
    face_index: int = Field(..., description="Index of detected face")
    label: str = Field(..., description="Predicted class ('With Mask' or 'Without Mask')")
    is_wearing_mask: bool = Field(..., description="True if face mask detected, False otherwise")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities breakdown")
    box: Optional[BoundingBox] = Field(None, description="Bounding box if face detected")

class PredictionResponse(BaseModel):
    success: bool = Field(True, description="Indicates if inference succeeded")
    primary_label: str = Field(..., description="Dominant classification result")
    is_wearing_mask: bool = Field(..., description="True if face mask is compliant")
    confidence: float = Field(..., description="Overall confidence score (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    face_count: int = Field(0, description="Total faces detected in the image")
    faces: List[FacePrediction] = Field(default_factory=list, description="Per-face detection and classification details")
    annotated_image: Optional[str] = Field(None, description="Base64 data URL of image with bounding boxes drawn")
    inference_time_ms: float = Field(..., description="Total processing & inference time in milliseconds")
    message: str = Field(..., description="Human-readable result summary")

class Base64ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image string or data URL (data:image/jpeg;base64,...)")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    version: str
    architecture: str

class ModelInfoResponse(BaseModel):
    app_name: str
    version: str
    classes: Dict[int, str]
    input_resolution: str
    metrics: Dict[str, object]
