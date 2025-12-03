"""
YOLO Test 1 API Routes
Endpoints for YOLO-based pose detection
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
from pathlib import Path
import shutil
import sys
from datetime import datetime

# Add yolo_test1 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "yolo_test1"))

router = APIRouter()

# Global variables
pose_estimator = None
config = None

def initialize_model():
    """Initialize YOLO Test1 model"""
    global pose_estimator, config
    try:
        from yolo_test1.pose_estimator import YOLOv8PoseEstimator
        from config import Config
        
        config = Config()
        pose_estimator = YOLOv8PoseEstimator(config)
        return True
    except Exception as e:
        print(f"❌ Error loading YOLO Test1: {e}")
        return False

# Initialize on startup
initialize_model()

@router.post("/predict")
async def predict_yolo1(file: UploadFile = File(...)):
    """
    YOLO Test1 pose prediction
    
    - **file**: Image file (jpg, png)
    """
    if pose_estimator is None:
        if not initialize_model():
            raise HTTPException(
                status_code=500,
                detail="YOLO Test1 model not loaded"
            )
    
    try:
        # Create directories
        input_dir = Path("yolo_test1/input")
        output_dir = Path("yolo_test1/output")
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = input_dir / filename
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process
        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        result_img, data = pose_estimator.estimate(img)
        
        # Save result
        output_filename = f"result_{filename}"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), result_img)
        
        return JSONResponse(content={
            "success": True,
            "message": "YOLO Test1 prediction completed",
            "data": data if data else {},
            "output_file": str(output_path),
            "download_url": f"/api/yolo-test1/result/{output_filename}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        await file.close()

@router.get("/result/{filename}")
async def get_result(filename: str):
    """Download result image"""
    file_path = Path("yolo_test1/output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)

@router.get("/info")
async def get_info():
    """Get YOLO Test1 info"""
    return {
        "project": "YOLO Test 1",
        "description": "YOLO-based pose estimation",
        "models": ["yolov8n-pose.pt", "yolov8x-pose.pt"],
        "status": "active" if pose_estimator else "inactive",
        "endpoints": {
            "predict": "/api/yolo-test1/predict",
            "result": "/api/yolo-test1/result/{filename}",
            "info": "/api/yolo-test1/info"
        }
    }