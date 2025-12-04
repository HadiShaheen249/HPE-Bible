# 🎯 HPE Bible - Human Pose Estimation API

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Professional REST API for Advanced Human Pose Estimation**

[Features](#-features) • [Installation](#️-installation) • [Usage](#-usage) • [API Docs](#-api-endpoints) • [Models](#-models)

</div>

---

## 📖 Overview

HPE Bible is a comprehensive FastAPI-based solution for human pose estimation, offering **two specialized approaches** optimized for different scenarios:

### 🔲 YOLO Test 1 - Tiled Pose Estimation
**Perfect for far-view scenes like sports stadiums and large areas**

- 🧩 **Intelligent Tiling**: Automatically divides images/videos into a configurable grid (2×2, 3×3, etc.)
- 🔍 **Long-Distance Detection**: Detects poses in distant subjects that standard models struggle with
- ⚽ **Sports Optimized**: Ideal for football matches, basketball games, and wide-angle surveillance
- 🎯 **Smart Merging**: Seamlessly combines results from all tiles with overlap handling
- 📐 **Adaptive Processing**: Each tile is processed at full resolution for maximum detail

**Why Tiling?**
When subjects are far from the camera, they occupy only a few pixels in the full image. By splitting the image into tiles, each section is processed at higher effective resolution, dramatically improving detection accuracy for distant subjects.

**Use Cases:**
- 🏟️ Sports field analysis
- 🎥 Crowd monitoring
- 📹 Surveillance footage
- 🏃 Marathon tracking

---

### 🎯 YOLO Test 2 - Two-Stage Pose Estimation
**High-accuracy pose estimation with dual-model pipeline**

- 🔍 **Stage 1 - Person Detection**: YOLOv8 object detection model identifies and localizes all persons
- 🧍 **Stage 2 - Pose Estimation**: YOLOv8-Pose model processes each detected person individually
- ✂️ **Smart Cropping**: Extracts tight bounding boxes around each person for optimal pose detection
- ⚙️ **Independent Scaling**: Use different model sizes for detection (fast) and pose (accurate)
- 📊 **Higher Precision**: Two-stage approach reduces false positives and improves keypoint accuracy

**The Two-Stage Advantage:**
1. **Detection Model** focuses solely on finding people → Faster and more efficient
2. **Pose Model** works on cropped, centered images → Better keypoint localization
3. **Flexible Resources** - Use small detection + large pose models for optimal performance

**Use Cases:**
- 👥 Group photos
- 🎬 Action recognition
- 🏋️ Fitness tracking
- 🕺 Dance analysis
- 🤸 Sports pose analysis

---

## ✨ Features

### 🚀 Core Capabilities
- **⚡ High Performance** - Built on FastAPI with async/await support
- **🤖 Multiple Models** - YOLOv8 variants: nano, small, medium, large, xlarge
- **🎭 Dual Strategies** - Choose between tiled or two-stage approaches
- **📸 Multi-Format** - Images (JPG, PNG, BMP, WebP) and Videos (MP4, AVI, MOV, MKV)
- **💻 Hardware Flexible** - Optimized for CPU, CUDA (NVIDIA), and MPS (Apple Silicon)

### 🎨 User Experience
- **📊 Interactive Docs** - Auto-generated Swagger UI and ReDoc
- **🎮 Model Playground** - Web interface to test models with live preview
- **📈 Real-time Progress** - Processing status and completion tracking
- **🖼️ Visual Results** - Annotated outputs with keypoints, skeletons, and bounding boxes

### 📦 Output Formats
- **🎨 Annotated Media** - Visual results with drawn pose overlays
- **📋 JSON Export** - Structured data with coordinates and confidence scores
- **📊 CSV Reports** - Tabular format for analysis in Excel/Python
- **📈 Statistics** - Per-frame/per-person analytics

---

## 🛠️ Installation

### Prerequisites

```bash
✓ Python 3.8 or higher
✓ pip (Python package manager)
✓ 4GB+ RAM (8GB recommended)
✓ ~2GB disk space for models
✓ CUDA Toolkit 11.8+ (optional, for GPU acceleration)