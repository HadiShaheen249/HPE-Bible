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

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
YOLO_TEST1_DIR = BASE_DIR / "yolo_test1"

# Add to Python path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Create router
router = APIRouter()

# Global variable for model
pose_estimator = None

def initialize_model():
    """Initialize YOLO Test1 model"""
    global pose_estimator
    
    try:
        print(f"📁 Loading YOLO Test1 from: {YOLO_TEST1_DIR}")
        
        # Import using full path (better approach)
        import yolo_test1.pose_estimator as pe_module
        
        # Create instance with correct parameters
        pose_estimator = pe_module.PoseEstimator(
            model_name="yolov8n-pose.pt",
            conf_threshold=0.5
        )
        
        print("✅ YOLO Test1 model loaded successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"   Make sure yolo_test1/pose_estimator.py exists")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ Error loading YOLO Test1: {e}")
        import traceback
        traceback.print_exc()
        return False

# Try to initialize on module load
print(f"🔄 Initializing YOLO Test1...")
initialize_model()

@router.post("/predict")
async def predict_yolo1(file: UploadFile = File(...)):
    """
    YOLO Test1 pose prediction
    
    - **file**: Image file (jpg, png)
    """
    global pose_estimator
    
    if pose_estimator is None:
        if not initialize_model():
            raise HTTPException(
                status_code=503,
                detail="YOLO Test1 model not loaded. Check server logs."
            )
    
    try:
        # Create directories
        input_dir = YOLO_TEST1_DIR / "input"
        output_dir = YOLO_TEST1_DIR / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = input_dir / filename
        
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"📥 Processing: {filename}")
        
        # Read image
        img = cv2.imread(str(input_path))
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process with pose estimator
        result = pose_estimator.estimate(img)
        
        # Handle different return types
        if isinstance(result, tuple):
            if len(result) >= 2:
                result_img, data = result[0], result[1]
            else:
                result_img = result[0]
                data = None
        else:
            result_img = img
            data = result
        
        # Save result image
        output_filename = f"result_{filename}"
        output_path = output_dir / output_filename
        
        if result_img is not None and isinstance(result_img, np.ndarray):
            cv2.imwrite(str(output_path), result_img)
            print(f"✅ Saved result: {output_filename}")
        else:
            cv2.imwrite(str(output_path), img)
            print(f"⚠️  Saved original image (no result image)")
        
        return JSONResponse(content={
            "success": True,
            "message": "YOLO Test1 prediction completed",
            "input_file": str(input_path),
            "output_file": str(output_path),
            "data": str(data) if data is not None else "No data",
            "download_url": f"/api/yolo-test1/result/{output_filename}"
        })
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/result/{filename}")
async def get_result(filename: str):
    """Download result image"""
    file_path = YOLO_TEST1_DIR / "output" / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"File not found: {filename}"
        )
    
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=filename
    )

@router.get("/info")
async def get_info():
    """Get YOLO Test1 info"""
    model_path = YOLO_TEST1_DIR / "models" / "yolov8n-pose.pt"
    pose_estimator_file = YOLO_TEST1_DIR / "pose_estimator.py"
    
    return {
        "project": "YOLO Test 1",
        "description": "YOLO-based pose estimation",
        "status": "active" if pose_estimator is not None else "inactive",
        "model_loaded": pose_estimator is not None,
        "paths": {
            "project_dir": str(YOLO_TEST1_DIR),
            "model_path": str(model_path),
            "model_exists": model_path.exists(),
            "pose_estimator_file": str(pose_estimator_file),
            "pose_estimator_exists": pose_estimator_file.exists()
        },
        "models": ["yolov8n-pose.pt", "yolov8x-pose.pt"],
        "endpoints": {
            "predict": "/api/yolo-test1/predict",
            "result": "/api/yolo-test1/result/{filename}",
            "info": "/api/yolo-test1/info",
            "reload": "/api/yolo-test1/reload"
        }
    }

@router.post("/reload")
async def reload_model():
    """Reload the model"""
    success = initialize_model()
    return {
        "success": success,
        "message": "Model reloaded successfully" if success else "Failed to reload model"
    }