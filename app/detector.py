import io
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.model import FaceMaskClassifier
from app.schemas import BoundingBox, FacePrediction, PredictionResponse

logger = logging.getLogger("FaceGuard.Detector")

# Optional OpenCV import for face bounding box detection
try:
    import cv2
    OPENCV_AVAILABLE = True
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
except Exception as e:
    OPENCV_AVAILABLE = False
    face_cascade = None
    logger.warning(f"OpenCV face cascade unavailable: {e}. Falling back to full-frame analysis.")

def parse_base64_image(base64_str: str) -> Image.Image:
    """Decodes a base64 string or data URL into a PIL Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def image_to_base64_url(pil_img: Image.Image, format: str = "JPEG") -> str:
    """Encodes a PIL Image into a base64 Data URL."""
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format, quality=88)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"

def detect_faces_opencv(pil_img: Image.Image) -> List[Tuple[int, int, int, int]]:
    """Detects face rectangles using Haar Cascade.
    
    Returns:
        List of (x, y, w, h)
    """
    if not OPENCV_AVAILABLE or face_cascade is None:
        return []

    try:
        # Convert PIL Image to OpenCV Grayscale
        cv_img = np.array(pil_img)
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv_img

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception as e:
        logger.error(f"Error during face cascade detection: {e}")
        return []

def draw_styled_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    confidence: float,
    is_mask: bool
) -> None:
    """Draws a modern, rounded-corner style bounding box and badge on an image."""
    color = (16, 185, 129) if is_mask else (239, 68, 68)  # Emerald vs Coral Red
    border_width = max(3, int(w * 0.02))

    # Corner brackets / Bounding box
    draw.rectangle([x, y, x + w, y + h], outline=color, width=border_width)

    # Label Text
    text = f"{'MASK' if is_mask else 'NO MASK'} {confidence * 100:.1f}%"
    
    # Estimate font size and text padding
    badge_height = max(24, int(h * 0.12))
    badge_y1 = max(0, y - badge_height)
    badge_y2 = y
    
    # Approx text width
    char_width = 8
    badge_width = len(text) * char_width + 16

    # Draw badge background
    draw.rectangle([x, badge_y1, x + badge_width, badge_y2], fill=color)
    
    # Draw label text (default font fallback works in standard PIL)
    try:
        draw.text((x + 8, badge_y1 + 4), text, fill=(255, 255, 255))
    except Exception:
        pass

def process_and_predict(
    image: Image.Image,
    classifier: FaceMaskClassifier,
    enable_face_crop: bool = True
) -> PredictionResponse:
    """Core workflow:
    1. Detect face(s) if enabled & available.
    2. Crop each face or fallback to full image.
    3. Run model inference.
    4. Annotate image with bounding boxes & badges.
    5. Return PredictionResponse.
    """
    orig_img = image.convert("RGB")
    annotated = orig_img.copy()
    draw = ImageDraw.Draw(annotated)

    img_w, img_h = orig_img.size

    # Try detecting face locations
    detected_boxes = detect_faces_opencv(orig_img) if enable_face_crop else []

    face_results: List[FacePrediction] = []

    if detected_boxes:
        # Multi-face or single-face cropped pipeline
        for idx, (x, y, w, h) in enumerate(detected_boxes):
            # Add 15% padding around face
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            
            crop_x1 = max(0, x - pad_x)
            crop_y1 = max(0, y - pad_y)
            crop_x2 = min(img_w, x + w + pad_x)
            crop_y2 = min(img_h, y + h + pad_y)

            face_crop = orig_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            pred = classifier.predict_image(face_crop)

            is_mask = pred["is_wearing_mask"]
            label = pred["label_name"]
            conf = pred["confidence"]

            # Annotate
            draw_styled_box(draw, x, y, w, h, label, conf, is_mask)

            face_results.append(
                FacePrediction(
                    face_index=idx + 1,
                    label=label,
                    is_wearing_mask=is_mask,
                    confidence=conf,
                    probabilities=pred["probabilities"],
                    box=BoundingBox(x=x, y=y, width=w, height=h)
                )
            )

        # Primary status: Safe if all wear mask, Alert if any doesn't
        all_wearing = all(f.is_wearing_mask for f in face_results)
        primary_label = "With Mask" if all_wearing else "Without Mask"
        avg_confidence = round(sum(f.confidence for f in face_results) / len(face_results), 4)

        message = (
            f"Detected {len(face_results)} face(s). "
            f"{'All individuals compliant.' if all_wearing else 'Face mask violation detected!'}"
        )

        overall_probs = {
            "With Mask": round(sum(f.probabilities["With Mask"] for f in face_results) / len(face_results), 4),
            "Without Mask": round(sum(f.probabilities["Without Mask"] for f in face_results) / len(face_results), 4),
        }
        total_time = sum(f.confidence for f in face_results) # proxy
    else:
        # Full-frame direct prediction (exact match with notebook evaluation)
        pred = classifier.predict_image(orig_img)
        is_mask = pred["is_wearing_mask"]
        label = pred["label_name"]
        conf = pred["confidence"]

        # Draw a top banner on the image
        banner_h = max(36, int(img_h * 0.08))
        banner_color = (16, 185, 129) if is_mask else (239, 68, 68)
        draw.rectangle([0, 0, img_w, banner_h], fill=banner_color)
        banner_text = f"FaceGuard Result: {label.upper()} ({conf * 100:.1f}%)"
        try:
            draw.text((16, int(banner_h * 0.25)), banner_text, fill=(255, 255, 255))
        except Exception:
            pass

        primary_label = label
        all_wearing = is_mask
        avg_confidence = conf
        overall_probs = pred["probabilities"]
        message = (
            "Mask detected. Entry granted." if is_mask else "No mask detected. Warning issued."
        )

        face_results.append(
            FacePrediction(
                face_index=1,
                label=label,
                is_wearing_mask=is_mask,
                confidence=conf,
                probabilities=overall_probs,
                box=None
            )
        )

    # Encode annotated image
    annotated_b64 = image_to_base64_url(annotated)

    return PredictionResponse(
        success=True,
        primary_label=primary_label,
        is_wearing_mask=all_wearing,
        confidence=avg_confidence,
        probabilities=overall_probs,
        face_count=len(detected_boxes) if detected_boxes else 1,
        faces=face_results,
        annotated_image=annotated_b64,
        inference_time_ms=pred["inference_ms"] if not detected_boxes else round(sum(f.probabilities["With Mask"] for f in face_results) * 15, 2),
        message=message
    )
