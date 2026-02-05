# FaceGuard-CNN

AI-powered Face Mask Detection System.

## Overview
Deep learning project for detecting face mask compliance using CNN.

### Dataset
Using Kaggle Face Mask Dataset (7,553 images).
- With Mask: 3,725 images
- Without Mask: 3,828 images

### Pipeline
- Kaggle API download and zip extraction configured.
- Verified RGB image formatting across dataset.
- Label mapping: 0 -> Without Mask, 1 -> With Mask.

### Preprocessing
- PIL Image RGB conversion pipeline.
- Image resizing standardized to (128, 128).
- Normalization to [0.0, 1.0] float32 array.

### Splitting
- 80/20 train/test split with stratified sampling.
- Verified zero data leakage between splits.
- Total training samples: 6,042, Test samples: 1,511.
