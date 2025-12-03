# 🎯 HPE Bible - Human Pose Estimation API

Unified FastAPI for three pose estimation projects:
- **RTMPose**: Real-time multi-person pose estimation
- **YOLO Test 1**: YOLO-based pose detection  
- **YOLO Test 2**: Advanced YOLO detection with multi-model support

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Models](#models)

## ✨ Features

- 🚀 **Fast API** - High-performance REST API
- 🤖 **Multiple Models** - RTMPose + YOLO variants
- 📸 **Image Processing** - Upload and process images
- 🎥 **Video Support** - Video processing (coming soon)
- 📊 **Interactive Docs** - Swagger UI & ReDoc
- 🔄 **Hot Reload** - Development mode with auto-reload

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Steps

```bash
# Clone the repository
git clone https://github.com/HadiShaheen249/HPE-Bible-API.git
cd HPE-Bible-API

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt