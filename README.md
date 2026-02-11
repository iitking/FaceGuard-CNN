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

### Augmentation
- Random horizontal flip enabled.
- Random rotation 8% and random zoom 10%.
- ImageDataGenerator parameters tuned with shear and shift.

### Architecture
- Input shape: (128, 128, 3).
- Conv Block 1: Conv2D(32, (3,3), activation='relu', padding='same').
- Conv Block 1: MaxPooling2D((2,2)).
- Conv Block 1: BatchNormalization layer added.
- Conv Block 2: Conv2D(64, (3,3), padding='same') + BatchNorm + MaxPool.
- Spatial reduction: 128x128 -> 64x64 -> 32x32.
- Conv Block 3: Conv2D(128, (3,3), padding='same').
- Conv Block 3: BatchNormalization and MaxPooling2D.
- Validated receptive field expansion.
- Conv Block 4: Conv2D(256, (3,3), padding='same').
- Conv Block 4: BatchNormalization added.
- Conv Block 4: MaxPooling2D -> final feature maps at 8x8x256.
- Head: GlobalAveragePooling2D replaces flatten to prevent overfitting.
