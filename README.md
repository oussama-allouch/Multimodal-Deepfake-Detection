<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Deepfake%20Detection-blueviolet?style=for-the-badge&logo=tensorflow" alt="Project Badge"/>
</p>

<h1 align="center">🛡️ DeepFake Detector</h1>

<p align="center">
  <strong>AI-powered deepfake detection system for images and videos</strong><br/>
  Built with CNN + Attention Mechanisms · Flask API · Next.js Interface
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white" alt="Next.js"/>
  <img src="https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Download Models](#download-models)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Docker Deployment](#docker-deployment)
- [Tech Stack](#tech-stack)
- [Authors](#authors)

---

## 🔍 Overview

**DeepFake Detector** is a full-stack application that leverages deep learning to detect AI-manipulated media (deepfakes) in both **images** and **videos**. The system uses a custom CNN architecture enhanced with attention mechanisms (Squeeze-and-Excitation, Spatial Attention, Temporal Attention) to achieve high-accuracy detection.

The application provides an intuitive web interface where users can upload images or videos and receive real-time analysis results with confidence scores, frame-by-frame analysis (for videos), and visual explanations.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                      │
│  ┌──────────┐  ┌────────────┐  ┌─────────────────────┐  │
│  │   Hero   │  │   Upload   │  │   Results Display   │  │
│  │ Section  │  │  Section   │  │  (Image + Video)    │  │
│  └──────────┘  └─────┬──────┘  └──────────▲──────────┘  │
│                      │                     │             │
│                 ┌────▼─────────────────────┴───┐        │
│                 │     Next.js API Routes       │        │
│                 │  /api/predict (images)        │        │
│                 │  /api/predict-video (videos)  │        │
│                 └────────────┬─────────────────┘        │
└──────────────────────────────┼──────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────┐
│                    Flask Backend API                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  /predict   │  │/predict-video│  │   /health     │  │
│  │  (images)   │  │  (videos)    │  │   /model-info │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────────┘  │
│         │                │                              │
│  ┌──────▼────────────────▼──────────────────────────┐   │
│  │              Model Handler                        │   │
│  │  ┌────────────────┐  ┌─────────────────────────┐ │   │
│  │  │ DeepfakeDetector│  │ VideoDeepfakeDetector  │ │   │
│  │  │ (images_model)  │  │ (videos_model_final)   │ │   │
│  │  └────────────────┘  └─────────────────────────┘ │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- **🖼️ Image Detection** — Upload single or multiple images for deepfake analysis
- **🎬 Video Detection** — Frame-by-frame video analysis with temporal attention
- **📊 Detailed Results** — Confidence scores, fake/real probabilities, and visualizations
- **🔍 Frame Analysis** — Per-frame breakdown for video predictions
- **⚡ Real-time Processing** — Fast inference with optimized preprocessing
- **🎨 Modern UI** — Responsive dark-themed interface with glassmorphism effects
- **🐳 Docker Support** — One-command deployment with Docker Compose
- **📖 Documentation** — Built-in documentation page explaining the model

---

## 🧠 Model Details

### Image Model (`images_model.keras` — 18 MB)

| Property        | Value                              |
|-----------------|------------------------------------|
| Architecture    | Custom CNN + SE Block + Spatial Attention |
| Input Shape     | `128 × 128 × 3` (RGB)             |
| Output          | P(Real) — sigmoid activation       |
| Loss Function   | Focal Binary Cross-Entropy         |
| Label Mapping   | `Fake: 0`, `Real: 1`              |

### Video Model (`videos_model_final.keras` — 168 MB)

| Property        | Value                              |
|-----------------|------------------------------------|
| Architecture    | CNN + BiLSTM + Temporal Attention  |
| Input           | Sequence of extracted frames       |
| Frame Sampling  | 1 frame/second (configurable)      |
| Max Frames      | 300 frames per video               |
| Output          | Aggregated prediction with per-frame scores |

### Custom Layers

- **SEBlock** — Squeeze-and-Excitation: channel-wise attention recalibration
- **SpatialAttention** — Highlights spatially important regions (manipulation artifacts)
- **TemporalAttention** — Additive attention over BiLSTM outputs for temporal consistency
- **FocalBCE** — Focal loss to handle class imbalance during training

---

## 📁 Project Structure

```
version_Final_DF_Detection/
├── app/                          # Next.js pages & API routes
│   ├── api/
│   │   ├── predict/              # Image prediction proxy
│   │   └── predict-video/        # Video prediction proxy
│   ├── about/                    # About page
│   ├── documentation/            # Documentation page
│   ├── layout.tsx                # Root layout
│   ├── page.tsx                  # Main page
│   └── globals.css               # Global styles
│
├── backend/                      # Flask API server
│   ├── models/                   # ⚠️ Model files (see Download section)
│   │   ├── images_model.keras
│   │   └── videos_model_final.keras
│   ├── uploads/                  # Temporary upload storage
│   ├── main.py                   # Flask app & API endpoints
│   ├── model_handler.py          # Model loading & inference logic
│   ├── config.py                 # Backend configuration
│   ├── requirements.txt          # Python dependencies
│   ├── test_api.py               # API tests
│   └── wsgi.py                   # WSGI entry point
│
├── components/                   # React components
│   ├── ui/                       # shadcn/ui components
│   ├── hero.tsx                  # Landing hero section
│   ├── features.tsx              # Features section
│   ├── how-it-works.tsx          # How it works section
│   ├── navbar.tsx                # Navigation bar
│   ├── upload-section.tsx        # File upload component
│   ├── results-display.tsx       # Image results component
│   └── video-results-display.tsx # Video results component
│
├── config/                       # Frontend configuration
├── hooks/                        # Custom React hooks
├── lib/                          # Utility functions
├── scripts/                      # Diagnostic & test scripts
├── styles/                       # Additional stylesheets
│
├── docker-compose.yml            # Docker orchestration
├── frontend.Dockerfile           # Frontend container
├── package.json                  # Node.js dependencies
├── tailwind.config.ts            # Tailwind CSS config
├── tsconfig.json                 # TypeScript config
└── README.md                     # This file
```

---

## 🚀 Installation

### Prerequisites

- **Node.js** ≥ 20.x
- **Python** ≥ 3.10
- **pip** (Python package manager)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/DeepFake-Detector.git
cd DeepFake-Detector
```

### 2. Install Frontend dependencies

```bash
npm install
```

### 3. Install Backend dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Download the trained models

> ⚠️ **The model files are too large for GitHub.** Follow the instructions below to download them.

---

## 📥 Download Models

The pre-trained model files are hosted on **Google Drive** due to their size.

| Model | Size | Google Drive Link |
|-------|------|-------------------|
| `images_model.keras` | 18 MB | [📥 Download](https://drive.google.com/YOUR_LINK_HERE) |
| `videos_model_final.keras` | 168 MB | [📥 Download](https://drive.google.com/YOUR_LINK_HERE) |

### Instructions

1. Download both model files from the links above
2. Place them in the `backend/models/` directory:

```
backend/
└── models/
    ├── images_model.keras      ← Place here
    └── videos_model_final.keras ← Place here
```

3. Verify the models are correctly placed:

```bash
ls backend/models/
# Expected output:
# images_model.keras
# videos_model_final.keras
```

> **Note:** The application will not start properly without the model files. The `/health` endpoint will report model loading status.

---

## ▶️ Usage

### Start the Backend (Flask API)

```bash
cd backend
python main.py
```

The API server starts on `http://localhost:8000`

### Start the Frontend (Next.js)

In a separate terminal:

```bash
npm run dev
```

The web interface opens on `http://localhost:3000`

### How to Use

1. Open `http://localhost:3000` in your browser
2. Choose between **Image** or **Video** detection
3. Upload your file(s) via drag & drop or file picker
4. View the analysis results with confidence scores

---

## 🔌 API Endpoints

### `GET /health`
Health check endpoint. Returns model loading status.

```json
{
  "status": "healthy",
  "image_model_loaded": true,
  "video_model_loaded": true
}
```

### `POST /predict`
Analyze images for deepfake detection.

- **Content-Type:** `multipart/form-data`
- **Field:** `files` (one or more image files)
- **Supported formats:** PNG, JPG, JPEG, GIF, BMP, TIFF

```json
{
  "predictions": [
    {
      "filename": "image.jpg",
      "prediction": "Fake",
      "confidence": 0.94,
      "fake_score": 0.94,
      "real_score": 0.06
    }
  ]
}
```

### `POST /predict-video`
Analyze videos with frame-by-frame detection.

- **Content-Type:** `multipart/form-data`
- **Field:** `file` (single video file)
- **Supported formats:** MP4, AVI, MOV, MKV, FLV, WMV, WEBM
- **Max size:** 500 MB

```json
{
  "filename": "video.mp4",
  "is_video": true,
  "prediction": "Fake",
  "confidence": 0.87,
  "frame_predictions": [...],
  "aggregated": {
    "prediction": "Fake",
    "fake_score": 0.87,
    "confidence": 0.87
  }
}
```

---

## 🐳 Docker Deployment

Build and run the entire stack with Docker Compose:

```bash
docker-compose up --build
```

This starts:
- **Backend** on port `8000`
- **Frontend** on port `3000`

To stop:

```bash
docker-compose down
```

---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| Next.js 16 | React framework with API routes |
| TypeScript | Type-safe development |
| Tailwind CSS | Utility-first styling |
| shadcn/ui | UI component library |
| Lucide React | Icon library |
| Recharts | Data visualization |

### Backend
| Technology | Purpose |
|------------|---------|
| Flask 3.1 | Lightweight Python web framework |
| TensorFlow 2.19 | Deep learning framework |
| Keras | High-level neural network API |
| OpenCV | Image/video processing |
| scikit-learn | Evaluation metrics |
| Pillow | Image manipulation |
| Gunicorn | Production WSGI server |

### Infrastructure
| Technology | Purpose |
|------------|---------|
| Docker | Containerization |
| Docker Compose | Multi-container orchestration |

---

## 📝 Configuration

Key configuration options in [`backend/config.py`](backend/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKEND_PORT` | 8000 | API server port |
| `MAX_FILE_SIZE` | 10 MB | Max image upload size |
| `MAX_VIDEO_SIZE` | 500 MB | Max video upload size |
| `VIDEO_FRAME_SAMPLING_RATE` | 1 Hz | Frames extracted per second |
| `VIDEO_MAX_FRAMES` | 300 | Maximum frames to process |
| `VIDEO_PROCESSING_TIMEOUT` | 300s | Video processing timeout |

---

## 👥 Authors

- **Oussama Allouch** — *AISD Master — Machine Learning & Deep Learning*

---

## 📄 License

This project was developed as part of the **Deep Learning** course in the **AISD Master's program** (S1 — Machine Learning 1).

---

<p align="center">
  <strong>⭐ If you found this project useful, please give it a star! ⭐</strong>
</p>
