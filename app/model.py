import os
import sys
import logging
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

from app.config import (
    MODEL_PATH,
    FALLBACK_MODEL_PATH,
    IMAGE_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LABEL_MAP
)

logger = logging.getLogger("FaceGuard.Model")

class FaceMaskClassifier:
    """Wrapper for the FaceGuard-CNN trained Keras model.
    
    Handles lazy loading, image preprocessing, and inference.
    """
    _instance: Optional["FaceMaskClassifier"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_PATH
        self.fallback_path = FALLBACK_MODEL_PATH
        self.model: Any = None
        self.is_loaded: bool = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "FaceMaskClassifier":
        """Singleton accessor for the model classifier."""
        if cls._instance is None:
            cls._instance = FaceMaskClassifier()
        return cls._instance

    def _load_model(self) -> None:
        """Attempt to load the Keras/TensorFlow model from disk."""
        target_path = self.model_path if self.model_path.exists() else self.fallback_path

        if not target_path.exists():
            logger.warning(
                f"Model file not found at {self.model_path} or {self.fallback_path}. "
                "Inference will not be available until the model file is placed."
            )
            return

        logger.info(f"Loading FaceGuard model from: {target_path}")

        try:
            # First try keras 3 / tensorflow keras
            try:
                import keras
                self.model = keras.models.load_model(str(target_path))
                self.is_loaded = True
                logger.info(f"Successfully loaded model via Keras: {target_path}")
                return
            except (ImportError, Exception) as ke:
                logger.debug(f"Direct keras load attempt: {ke}. Trying tensorflow.keras...")

            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(str(target_path))
                self.is_loaded = True
                logger.info(f"Successfully loaded model via tf.keras: {target_path}")
                return
            except (ImportError, Exception) as te:
                logger.debug(f"tf.keras load attempt: {te}. Trying pickle fallback...")

            # Fallback to pickle if .pkl is provided
            if target_path.suffix == ".pkl" or self.fallback_path.exists():
                pkl_target = self.fallback_path if self.fallback_path.exists() else target_path
                import pickle
                with open(pkl_target, "rb") as f:
                    self.model = pickle.load(f)
                self.is_loaded = True
                logger.info(f"Successfully loaded model via pickle: {pkl_target}")
                return

        except Exception as e:
            logger.error(f"Failed to load FaceGuard model: {e}", exc_info=True)
            self.model = None
            self.is_loaded = False

    def preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """Preprocesses a PIL Image matching the exact training pipeline:
        1. Convert to RGB
        2. Resize to 128x128
        3. Convert to float32 / 255.0
        4. Expand batch dimension -> shape (1, 128, 128, 3)
        """
        rgb_image = pil_image.convert("RGB")
        resized_image = rgb_image.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        img_array = np.array(resized_image, dtype=np.float32) / 255.0
        batch_array = np.expand_dims(img_array, axis=0)
        return batch_array

    def predict_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Runs inference on a single PIL Image.
        
        Returns:
            Dict containing:
                - label_idx (0 or 1)
                - label_name ("Without Mask" or "With Mask")
                - is_wearing_mask (bool)
                - confidence (float)
                - probabilities (Dict[str, float])
                - inference_ms (float)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "FaceGuard model is not loaded. Please ensure 'FaceGuard-CNN_model.keras' exists "
                "and TensorFlow/Keras is installed."
            )

        start_time = time.perf_counter()
        preprocessed = self.preprocess(pil_image)

        # Handle tensor or keras model predict
        raw_pred = self.model.predict(preprocessed, verbose=0)
        
        # Ensure numpy array
        if hasattr(raw_pred, "numpy"):
            raw_pred = raw_pred.numpy()

        probs = raw_pred[0]
        prob_without_mask = float(probs[0])
        prob_with_mask = float(probs[1])

        pred_idx = int(np.argmax(probs))
        label_name = LABEL_MAP.get(pred_idx, "Unknown")
        confidence = float(np.max(probs))
        is_wearing_mask = (pred_idx == 1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "label_idx": pred_idx,
            "label_name": label_name,
            "is_wearing_mask": is_wearing_mask,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Without Mask": round(prob_without_mask, 4),
                "With Mask": round(prob_with_mask, 4)
            },
            "inference_ms": round(elapsed_ms, 2)
        }
