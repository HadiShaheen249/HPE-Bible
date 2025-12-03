"""
YOLO Test 2 API Routes
Advanced YOLO detection with video support
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
import cv2
from pathlib import Path
import shutil
import sys
from datetime import datetime
import importlib.util

# Add yolo_test2 to path
BASE_DIR = Path(__file__).parent.parent.parent
YOLO_TEST2_DIR = BASE_DIR / "yolo_test2"

if str(YOLO_TEST2_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_TEST2_DIR))

router = APIRouter()

# Global variables
detector = None
config = None
Detector = None
Config = None

def load_modules():
    """Dynamically load detector and config modules"""
    global Detector, Config
    
    try:
        # Load detector.py
        detector_spec = importlib.util.spec_from_file_location(
            "detector", 
            YOLO_TEST2_DIR / "detector.py"
        )
        detector_module = importlib.util.module_from_spec(detector_spec)
        detector_spec.loader.exec_module(detector_module)
        Detector = detector_module.Detector
        
        # Load config.py
        config_spec = importlib.util.spec_from_file_location(
            "config_yolo2", 
            YOLO_TEST2_DIR / "config.py"
        )
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        Config = config_module.Config
        
        print("✅ YOLO Test2 modules loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading modules: {e}")
        return False

def initialize_model():
    """Initialize YOLO Test2 model"""
    global detector, config
    
    try:
        # Load modules first
        if not load_modules():
            return False
        
        # Initialize model
        config = Config()
        detector = Detector(config)
        print("✅ YOLO Test2 model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing YOLO Test2: {e}")
        print(f"📁 Looking in: {YOLO_TEST2_DIR}")
        return False

# Initialize
initialize_model()

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Object detection with YOLO Test2
    
    - **file**: Image file
    """
    if detector is None:
        if not initialize_model():
            raise HTTPException(
                status_code=500,
                detail="YOLO Test2 model not loaded. Check if detector.py exists."
            )
    
    try:
        # Directories
        input_dir = Path("yolo_test2/input")
        output_dir = Path("yolo_test2/output")
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = input_dir / filename
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Detect
        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        result_img, detections = detector.detect(img)
        
        # Save
        output_filename = f"result_{filename}"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), result_img)
        
        return JSONResponse(content={
            "success": True,
            "message": "Detection completed",
            "detections": detections if detections else [],
            "num_detections": len(detections) if detections else 0,
            "output_file": str(output_path),
            "download_url": f"/api/yolo-test2/result/{output_filename}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        await file.close()

@router.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    Process video with YOLO Test2
    
    - **file**: Video file (mp4, avi)
    """
    raise HTTPException(
        status_code=501,
        detail="Video processing not implemented yet. Coming soon!"
    )

@router.get("/result/{filename}")
async def get_result(filename: str):
    """Download result image"""
    file_path = Path("yolo_test2/output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)

@router.get("/info")
async def get_info():
    """Get YOLO Test2 info"""
    detector_exists = (YOLO_TEST2_DIR / "detector.py").exists()
    
    return {
        "project": "YOLO Test 2",
        "description": "Advanced YOLO object detection and pose estimation",
        "models": [
            "yolov8m.pt",
            "yolov8m-pose.pt",
            "yolov8x.pt",
            "yolov8x-pose.pt"
        ],
        "status": "active" if detector else "inactive",
        "detector_file_exists": detector_exists,
        "detector_path": str(YOLO_TEST2_DIR / "detector.py"),
        "features": ["Object Detection", "Pose Estimation", "Video Processing"],
        "endpoints": {
            "detect": "/api/yolo-test2/detect",
            "process_video": "/api/yolo-test2/process-video",
            "result": "/api/yolo-test2/result/{filename}",
            "info": "/api/yolo-test2/info"
        }
    }