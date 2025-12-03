"""
RTMPose API Routes
Endpoints for real-time pose estimation
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
from pathlib import Path
import shutil
import sys
from datetime import datetime

# Add RTMPose to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "RTMPose"))

router = APIRouter()

# Global variables for model
pose_estimator = None
config = None

def initialize_model():
    """Initialize RTMPose model"""
    global pose_estimator, config
    try:
        from RTMPose.pose_estimator import PoseEstimator
        from config import Config
        
        config = Config()
        pose_estimator = PoseEstimator(config)
        return True
    except Exception as e:
        print(f"❌ Error loading RTMPose: {e}")
        return False

# Try to initialize on startup
initialize_model()

@router.post("/predict")
async def predict_pose(file: UploadFile = File(...)):
    """
    Predict pose from uploaded image
    
    - **file**: Image file (jpg, png)
    
    Returns keypoints and result image path
    """
    if pose_estimator is None:
        if not initialize_model():
            raise HTTPException(
                status_code=500, 
                detail="RTMPose model not loaded"
            )
    
    try:
        # Create directories
        input_dir = Path("RTMPose/input")
        output_dir = Path("RTMPose/output")
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = input_dir / filename
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run prediction
        result_img, keypoints = pose_estimator.estimate(img)
        
        # Save result
        output_filename = f"result_{filename}"
        output_path = output_dir / output_filename
        cv2.imwrite(str(output_path), result_img)
        
        return JSONResponse(content={
            "success": True,
            "message": "Pose estimation completed",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "keypoints": keypoints.tolist() if keypoints is not None else [],
            "num_persons": len(keypoints) if keypoints is not None else 0,
            "download_url": f"/api/rtmpose/result/{output_filename}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        await file.close()

@router.get("/result/{filename}")
async def get_result(filename: str):
    """
    Download result image
    
    - **filename**: Result image filename
    """
    file_path = Path("RTMPose/output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg",
        filename=filename
    )

@router.get("/info")
async def get_info():
    """
    Get RTMPose model information
    """
    return {
        "project": "RTMPose",
        "description": "Real-time multi-person pose estimation",
        "status": "active" if pose_estimator is not None else "inactive",
        "model_loaded": pose_estimator is not None,
        "endpoints": {
            "predict": "/api/rtmpose/predict",
            "result": "/api/rtmpose/result/{filename}",
            "info": "/api/rtmpose/info"
        }
    }

@router.post("/reload")
async def reload_model():
    """
    Reload RTMPose model
    """
    success = initialize_model()
    return {
        "success": success,
        "message": "Model reloaded" if success else "Failed to reload model"
    }