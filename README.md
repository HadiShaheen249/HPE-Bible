# 🎯 HPE Bible - Human Pose Estimation API

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-orange.svg)
![RTMPose](https://img.shields.io/badge/RTMPose-MMPose-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Professional REST API for Advanced Human Pose Estimation**

[Features](#-features) • [Installation](#️-installation) • [Usage](#-usage) • [API Docs](#-api-endpoints) • [Models](#-models)

</div>

---

## 📖 Overview

HPE Bible is a comprehensive FastAPI-based solution for human pose estimation, offering **three specialized approaches** optimized for different scenarios:

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

### 🔥 RTMPose - Real-Time Multi-Person Pose Estimation (NEW!)
**State-of-the-art pose estimation powered by MMPose framework**

- ⚡ **Ultra-Fast Inference**: Optimized for real-time applications with minimal latency
- 🎯 **Superior Accuracy**: Achieves state-of-the-art results on COCO benchmark
- 🔄 **Multi-Person Tracking**: Integrated ByteTrack for consistent person ID across frames
- 🏗️ **SimCC Architecture**: Uses SimCC (Simple Coordinate Classification) for precise keypoint localization
- 📊 **133 Keypoints Support**: Full-body pose estimation including face, hands, and body (WholeBody model)

**RTMPose Highlights:**

| Feature | Specification |
|---------|---------------|
| 🎯 Model | RTMPose-m (Medium) |
| 📐 Keypoints | 17 (Body) / 133 (WholeBody) |
| ⚡ Speed | ~30+ FPS on GPU |
| 🎪 Backbone | CSPNeXt |
| 📏 Input Size | 256×192 / 384×288 |

**Why RTMPose?**
RTMPose represents the latest advancement in pose estimation from the MMPose team. It combines the speed of lightweight models with the accuracy of heavyweight ones, making it perfect for production deployments.

**Key Advantages:**
- 🚀 **3x Faster** than traditional top-down methods
- 📈 **Higher AP** (Average Precision) on COCO dataset
- 🔧 **Easy Integration** with existing pipelines
- 💪 **Robust** to occlusion and crowded scenes

**Use Cases:**
- 🎮 Real-time gaming & AR/VR
- 🏥 Medical rehabilitation tracking
- 🎭 Motion capture for animation
- 🛡️ Security & surveillance
- 🏋️ Professional sports analysis

---

## 🔄 Model Comparison

| Feature | YOLO Tiled | YOLO Two-Stage | RTMPose |
|---------|------------|----------------|---------|
| **Best For** | Far-view scenes | High precision | Real-time apps |
| **Speed** | ⚡⚡ | ⚡ | ⚡⚡⚡ |
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Multi-Person** | ✅ | ✅ | ✅ + Tracking |
| **Distant Subjects** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Crowded Scenes** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Real-Time** | ✅ | ⚠️ | ✅✅ |

---

## ✨ Features

### 🚀 Core Capabilities
- **⚡ High Performance** - Built on FastAPI with async/await support
- **🤖 Multiple Models** - YOLOv8 variants + RTMPose (nano to xlarge)
- **🎭 Triple Strategies** - Tiled, Two-Stage, or RTMPose approaches
- **📸 Multi-Format** - Images (JPG, PNG, BMP, WebP) and Videos (MP4, AVI, MOV, MKV)
- **💻 Hardware Flexible** - Optimized for CPU, CUDA (NVIDIA), and MPS (Apple Silicon)
- **🔄 Person Tracking** - ByteTrack integration for consistent IDs across video frames

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
✓ ~3GB disk space for models
✓ CUDA Toolkit 11.8+ (optional, for GPU acceleration)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/HadiShaheen249/HPE-Bible.git
cd HPE-Bible

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# For RTMPose (additional dependencies)
cd RTMPose
pip install -r requirements.txt
# OR use conda
conda env create -f environment.yml

🚀 Usage
Running the API
bash
# YOLO Test 1 - Tiled Approach
cd yolo_test1
python main.py

# YOLO Test 2 - Two-Stage Approach
cd yolo_test2
python main.py

# RTMPose - Real-Time Pose Estimation
cd RTMPose
python main.py
API will be available at:
🌐 Swagger UI: http://localhost:8000/docs
📚 ReDoc: http://localhost:8000/redoc
📊 API Endpoints
Endpoint	Method	Description
/health	GET	Health check
/process/image	POST	Process single image
/process/video	POST	Process video file
/models	GET	List available models
/config	GET/POST	View/Update configuration
🧠 Models
YOLOv8 Pose Models
Model	Size	Speed	Accuracy
yolov8n-pose	Nano	⚡⚡⚡⚡⚡	⭐⭐
yolov8s-pose	Small	⚡⚡⚡⚡	⭐⭐⭐
yolov8m-pose	Medium	⚡⚡⚡	⭐⭐⭐⭐
yolov8l-pose	Large	⚡⚡	⭐⭐⭐⭐⭐
yolov8x-pose	XLarge	⚡	⭐⭐⭐⭐⭐
RTMPose Models
Model	Keypoints	Input Size	AP (COCO)
rtmpose-t	17	256×192	68.5
rtmpose-s	17	256×192	72.2
rtmpose-m	17	256×192	75.8
rtmpose-l	17	256×192	76.5
rtmpose-m-wholebody	133	256×192	60.2
📁 Project Structure
text
HPE-Bible/
├── 📂 yolo_test1/          # Tiled pose estimation
│   ├── main.py
│   ├── config.py
│   ├── pose_estimator.py
│   └── requirements.txt
│
├── 📂 yolo_test2/          # Two-stage pose estimation
│   ├── main.py
│   ├── config.py
│   ├── detector.py
│   ├── pose_estimator.py
│   └── requirements.txt
│
├── 📂 RTMPose/             # RTMPose implementation
│   ├── main.py
│   ├── config.py
│   ├── pose_estimator.py
│   ├── byte_tracker.py     # Multi-person tracking
│   ├── environment.yml
│   └── requirements.txt
│
├── 📂 api/                 # Shared API components
├── 📄 config.py            # Global configuration
├── 📄 requirements.txt     # Main dependencies
└── 📄 README.md
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


🙏 Acknowledgments
Ultralytics for YOLOv8
MMPose for RTMPose
ByteTrack for multi-object tracking
Made with ❤️ by :
Hadi Shaheen
Mostafa Khaled
Marawan Sitten
Mohamed Salah
Mosa Mohamed

⭐ Star this repo if you find it useful!
