"""
YOLO Test 2 API Routes
Advanced YOLO detection
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
YOLO_TEST2_DIR = BASE_DIR / "yolo_test2"

# Add to Python path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Create router
router = APIRouter()

# Global variables
detector = None

def initialize_model():
    """Initialize YOLO Test2 model"""
    global detector
    
    try:
        print(f"📁 Loading YOLO Test2 from: {YOLO_TEST2_DIR}")
        
        # Check if detector.py exists
        detector_file = YOLO_TEST2_DIR / "detector.py"
        if not detector_file.exists():
            print(f"❌ detector.py not found at {detector_file}")
            return False
        
        # Import using full path
        import yolo_test2.detector as detector_module
        
        # Try to load config if exists
        config_file = YOLO_TEST2_DIR / "config.py"
        if config_file.exists():
            import yolo_test2.config as config_module
            config = config_module.Config()
            detector = detector_module.Detector(config)
        else:
            # Try without config
            print("⚠️  config.py not found, trying without config...")
            detector = detector_module.Detector()
        
        print("✅ YOLO Test2 model loaded successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"   Make sure yolo_test2/detector.py exists")
        import traceback
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"❌ Error loading YOLO Test2: {e}")
        import traceback
        traceback.print_exc()
        return False

# Try to initialize on module load
print(f"🔄 Initializing YOLO Test2...")
initialize_model()

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Object detection with YOLO Test2
    
    - **file**: Image file
    """
    global detector
    
    if detector is None:
        if not initialize_model():
            raise HTTPException(
                status_code=503,
                detail="YOLO Test2 detector not loaded. Check server logs."
            )
    
    try:
        # Create directories
        input_dir = YOLO_TEST2_DIR / "input"
        output_dir = YOLO_TEST2_DIR / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
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
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Detect
        result = detector.detect(img)
        
        # Handle result
        if isinstance(result, tuple):
            result_img, detections = result[0], result[1]
        else:
            result_img = img
            detections = result
        
        # Save
        output_filename = f"result_{filename}"
        output_path = output_dir / output_filename
        
        if result_img is not None and isinstance(result_img, np.ndarray):
            cv2.imwrite(str(output_path), result_img)
            print(f"✅ Saved result: {output_filename}")
        else:
            cv2.imwrite(str(output_path), img)
            print(f"⚠️  Saved original image")
        
        return JSONResponse(content={
            "success": True,
            "message": "Detection completed",
            "detections": str(detections) if detections else [],
            "output_file": str(output_path),
            "download_url": f"/api/yolo-test2/result/{output_filename}"
        })
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/result/{filename}")
async def get_result(filename: str):
    """Download result image"""
    file_path = YOLO_TEST2_DIR / "output" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=filename
    )

@router.get("/info")
async def get_info():
    """Get YOLO Test2 info"""
    detector_file = YOLO_TEST2_DIR / "detector.py"
    config_file = YOLO_TEST2_DIR / "config.py"
    
    return {
        "project": "YOLO Test 2",
        "description": "Advanced YOLO object detection",
        "status": "active" if detector else "inactive",
        "detector_loaded": detector is not None,
        "paths": {
            "project_dir": str(YOLO_TEST2_DIR),
            "detector_file": str(detector_file),
            "detector_exists": detector_file.exists(),
            "config_file": str(config_file),
            "config_exists": config_file.exists()
        },
        "models": ["yolov8m.pt", "yolov8m-pose.pt"],
        "endpoints": {
            "detect": "/api/yolo-test2/detect",
            "result": "/api/yolo-test2/result/{filename}",
            "info": "/api/yolo-test2/info"
        }
    }