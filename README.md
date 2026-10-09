# 🛡️ FaceGuard-CNN | AI Face Mask Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-98.28%25-10B981?style=for-the-badge&logo=target" alt="Accuracy">
  <img src="https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License">
</p>

---

## 🌟 Overview

**FaceGuard-CNN** is a high-accuracy, production-ready computer vision system engineered for real-time face mask detection and safety compliance surveillance.

Trained on the Kaggle Face Mask Dataset (7,553 labeled images), the custom **4-Block Convolutional Neural Network (CNN)** achieves **98.28% test accuracy** and **0.0548 cross-entropy loss** with an average inference latency of under **20 milliseconds**.

The project includes an interactive web dashboard with **live webcam monitoring**, **audio alarms**, **multi-face bounding box annotations**, and high-speed **FastAPI REST endpoints** for easy integration into CCTV streams, security gates, and mobile apps.

---

## ✨ Key Features

- 🧠 **High Accuracy Deep CNN**: 98.28% test accuracy with 4 convolutional blocks, Batch Normalization, and Dropout regularization.
- ⚡ **Ultra-Fast Inference**: Global Average Pooling and optimized feedforward pipeline (~18ms per inference).
- 📹 **Real-Time Webcam Surveillance**: Live camera feed with automated periodic scanning, FPS counter, and Web Audio API alerts on violations.
- 🎯 **Multi-Face Detection & Bounding Boxes**: Automatically detects and crops multiple individual faces in group photos using OpenCV Haar Cascades, annotating each face with colored compliance boxes.
- 🌐 **Modern Cyber UI Dashboard**: Single Page Application (SPA) built with pure vanilla HTML5, CSS3 glassmorphism, and responsive design (no external bulky JS frameworks).
- 🚀 **Production REST API**: Fully asynchronous FastAPI backend with automated OpenAPI / Swagger documentation at `/docs`.
- 🐳 **Docker & Cloud Ready**: Production `Dockerfile` and `docker-compose.yml` for instant deployment to Render, Railway, Hugging Face Spaces, or AWS.

---

## 🧠 Model Architecture & Performance

```
Input (128x128x3 RGB)
       │
       ▼
Data Augmentation (RandomFlip, RandomRotation 8%, RandomZoom 10%)
       │
       ▼
[Conv Block 1] ── Conv2D (32, 3x3) ── BatchNorm ── MaxPool2D (2x2)
       │
       ▼
[Conv Block 2] ── Conv2D (64, 3x3) ── BatchNorm ── MaxPool2D (2x2)
       │
       ▼
[Conv Block 3] ── Conv2D (128, 3x3) ── BatchNorm ── MaxPool2D (2x2)
       │
       ▼
[Conv Block 4] ── Conv2D (256, 3x3) ── BatchNorm ── MaxPool2D (2x2)
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
Dense (128, ReLU) ── BatchNorm ── Dropout (0.50)
       │
       ▼
Dense (64, ReLU) ── Dropout (0.30)
       │
       ▼
Dense (2, Softmax) ──► [Without Mask (0) | With Mask (1)]
```

### 📊 Training & Validation Metrics

| Metric | Score | Details |
|---|---|---|
| **Test Accuracy** | **98.28%** | Evaluated on unseen test split |
| **Test Loss** | **0.0548** | Sparse Categorical Cross-Entropy |
| **Input Shape** | `128 x 128 x 3` | Normalized to `[0.0, 1.0]` |
| **Training Epochs** | 30 | Adam Optimizer (`lr=0.001`) with Early Stopping |
| **Model Size** | `~5.2 MB` | Highly portable for edge and web serving |

---

## 📁 Project Structure

```bash
FaceGuard-CNN/
├── app/
│   ├── __init__.py           # Application package
│   ├── config.py             # Settings, paths, environment configuration
│   ├── model.py              # Model loader & inference wrapper
│   ├── detector.py           # Face detection, multi-face crop & annotation
│   ├── schemas.py            # Pydantic request/response data models
│   ├── main.py               # FastAPI router, CORS, lifespan & endpoints
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css     # Cyber-security dark theme & animations
│   │   └── js/
│   │       └── app.js        # Webcam streamer, audio synthesis & UI logic
│   └── templates/
│       └── index.html        # Interactive single-page web dashboard
├── tests/
│   ├── __init__.py
│   └── test_api.py           # Automated unit tests for API endpoints
├── FaceGuard-CNN_model.keras # Trained Keras model weights (Untouched)
├── FaceGuard-CNN_model.pkl   # Serialized model artifact (Untouched)
├── FaceGuard_CNN.ipynb       # Original Colab/Jupyter training notebook (Untouched)
├── run.py                    # Convenient CLI application runner
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production container specification
├── docker-compose.yml        # Docker compose configuration
├── .gitignore                # Git ignore rules
├── .dockerignore              # Docker build context exclusions
├── .env.example              # Environment variables template
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/FaceGuard-CNN.git
cd FaceGuard-CNN
```

### 2. Create a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for Apple Silicon (M1/M2/M3/M4 Mac users):**  
> `requirements.txt` automatically handles `tensorflow-macos` for hardware-accelerated inference.

### 4. Run the Application
```bash
python run.py
```

The application will start on **`http://localhost:8000`**!

- 🖥️ **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- 📖 **Interactive Swagger API**: [http://localhost:8000/docs](http://localhost:8000/docs)
- ❤️ **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Docker Deployment

Run FaceGuard-CNN in an isolated production container with one command:

### Using Docker Compose:
```bash
docker compose up --build
```

### Using standard Docker:
```bash
docker build -t faceguard-cnn .
docker run -p 8000:8000 faceguard-cnn
```

Visit `http://localhost:8000` in your browser.

---

## 📡 REST API Reference

### 1. Predict from Image File
**Endpoint**: `POST /api/predict`  
**Content-Type**: `multipart/form-data`  
**Query Parameters**: `detect_faces=true` (optional, default `true`)

#### Example cURL:
```bash
curl -X POST "http://localhost:8000/api/predict?detect_faces=true" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_photo.jpg"
```

#### Example Response:
```json
{
  "success": true,
  "primary_label": "With Mask",
  "is_wearing_mask": true,
  "confidence": 0.9852,
  "probabilities": {
    "With Mask": 0.9852,
    "Without Mask": 0.0148
  },
  "face_count": 1,
  "faces": [
    {
      "face_index": 1,
      "label": "With Mask",
      "is_wearing_mask": true,
      "confidence": 0.9852,
      "probabilities": {
        "With Mask": 0.9852,
        "Without Mask": 0.0148
      },
      "box": {
        "x": 120,
        "y": 80,
        "width": 140,
        "height": 160
      }
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "inference_time_ms": 18.5,
  "message": "Detected 1 face(s). All individuals compliant."
}
```

---

### 2. Predict from Base64 String (Webcam / Live Stream)
**Endpoint**: `POST /api/predict-base64`  
**Content-Type**: `application/json`

#### Example Python Request:
```python
import requests
import base64

with open("frame.jpg", "rb") as f:
    b64_str = base64.b64encode(f.read()).decode("utf-8")

payload = {"image": f"data:image/jpeg;base64,{b64_str}"}
response = requests.post("http://localhost:8000/api/predict-base64", json=payload)
print(response.json())
```

---

### 3. Service Health Check
**Endpoint**: `GET /health`

```bash
curl -X GET "http://localhost:8000/health"
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "FaceGuard-CNN_model.keras",
  "version": "1.0.0",
  "architecture": "Custom 4-Block CNN with Data Augmentation & BatchNorm"
}
```

---

## 🧪 Running Tests

Ensure all endpoints and modules are working as expected:

```bash
pytest tests/ -v
```

---

## 📤 Pushing to GitHub (Step-by-Step)

If you haven't uploaded this repository to GitHub yet, run these commands in your terminal:

```bash
# 1. Initialize git
git init

# 2. Add all project files
git add .

# 3. Create your first commit
git commit -m "feat: initial commit of FaceGuard-CNN with FastAPI and Web UI"

# 4. Rename default branch to main
git branch -M main

# 5. Add your GitHub repository remote (replace with your repo URL)
git remote add origin https://github.com/<YOUR_USERNAME>/FaceGuard-CNN.git

# 6. Push to GitHub!
git push -u origin main
```

---

## 👨‍💻 Author

<div align="center">

<a href="https://github.com/iitking">
  <img src="https://github.com/iitking.png" width="110" height="110" style="border-radius:50%" alt="Nivesh Kumar Meena" />
</a>

### **Nivesh Kumar Meena**

**AI Architect · MLOps Engineer** | B.Tech Electrical Engineering, **IIT Roorkee**

*Building agentic AI systems, RAG pipelines and production-ready ML.*

<a href="https://www.linkedin.com/in/nivesh-kumar-meena-a31465221/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
<a href="https://github.com/iitking"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" /></a>
<a href="mailto:niveshkr149@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" /></a>

<br/><br/>

⭐ **If you found this project useful, please give it a star!** ⭐

<sub>Open to AI/ML engineering opportunities and collaborations.</sub>

</div>

---

<div align="center">
  <sub>Made with ❤️ by <a href="https://github.com/iitking">Nivesh Kumar Meena</a> · © 2026 · MIT License</sub>
</div>
</p>
