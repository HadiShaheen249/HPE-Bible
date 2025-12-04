"""
YOLO Test 1 API Routes - Uses Original YOLOv8PoseEstimator
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add yolo_test1 to path
BASE_DIR = Path(__file__).parent.parent.parent
YOLO_TEST1_DIR = BASE_DIR / "yolo_test1"

if str(YOLO_TEST1_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_TEST1_DIR))

# Change to yolo_test1 directory for imports
original_cwd = os.getcwd()
os.chdir(YOLO_TEST1_DIR)

try:
    from yolo_test1.pose_estimator import YOLOv8PoseEstimator
    from yolo_test1.config import Config
finally:
    os.chdir(original_cwd)

router = APIRouter()
pose_estimators = {}  # Cache for different model sizes

def get_estimator(model_size='m'):
    """Get or create pose estimator of specified size"""
    global pose_estimators
    
    if model_size not in pose_estimators:
        try:
            model_name = f"yolov8{model_size}-pose.pt"
            print(f"🔄 Initializing YOLO Test1 with {model_name}")
            
            estimator = YOLOv8PoseEstimator(
                model_name=model_name,
                conf_threshold=0.5
            )
            
            pose_estimators[model_size] = estimator
            print(f"✅ YOLO Test1 ({model_size}) initialized successfully")
            return estimator
            
        except Exception as e:
            print(f"❌ Error initializing model {model_size}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return pose_estimators[model_size]

@router.post("/predict")
async def predict_yolo1(
    file: UploadFile = File(...),
    model_size: str = Form('m')
):
    """
    Pose estimation using original YOLOv8PoseEstimator
    model_size: n, s, m, l, x
    """
    estimator = get_estimator(model_size)
    if estimator is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Failed to load model size '{model_size}'"
        )
    
    try:
        input_dir = YOLO_TEST1_DIR / "input"
        output_dir = YOLO_TEST1_DIR / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = input_dir / filename
        
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        file_ext = input_path.suffix.lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']
        
        if is_video:
            # Process video using original method
            print(f"🎬 Processing video with YOLO Test1 ({model_size})...")
            
            output_filename = f"result_{timestamp}.mp4"
            output_path = output_dir / "videos" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use original predict_video method
            estimator.predict_video(
                video_path=str(input_path),
                save_result=True,
                output_path=str(output_path),
                show_live=False  # Don't show window in API
            )
            
            # Count frames (approximate)
            cap = cv2.VideoCapture(str(input_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return {
                "success": True,
                "message": "Video processing completed",
                "type": "video",
                "model_size": model_size,
                "frames_processed": frame_count,
                "download_url": f"/api/yolo-test1/result/videos/{output_filename}"
            }
        else:
            # Process image using original method
            print(f"📸 Processing image with YOLO Test1 ({model_size})...")
            
            output_filename = f"result_{filename}"
            output_path = output_dir / "images" / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # ✅ Use modified predict_image that returns counts
            result_frame, num_persons, total_keypoints = estimator.predict_image(
                image_path=str(input_path),
                save_result=True,
                output_path=str(output_path)
            )
            
            print(f"   ✅ Detected {num_persons} persons with {total_keypoints} total visible keypoints")
            
            return {
                "success": True,
                "message": "Pose estimation completed",
                "type": "image",
                "model_size": model_size,
                "num_persons": num_persons,  # ✅ العدد الصحيح من الـ tiles
                "total_visible_keypoints": total_keypoints,
                "tile_grid": list(Config.TILE_GRID),
                "download_url": f"/api/yolo-test1/result/images/{output_filename}"
            }
    
    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/result/{folder}/{filename}")
async def get_result(folder: str, filename: str):
    """Download result (images or videos)"""
    if folder not in ['images', 'videos']:
        raise HTTPException(status_code=400, detail="Invalid folder")
    
    file_path = YOLO_TEST1_DIR / "output" / folder / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    ext = file_path.suffix.lower()
    media_type = "video/mp4" if ext in ['.mp4', '.avi'] else "image/jpeg"
    
    return FileResponse(str(file_path), media_type=media_type, filename=filename)

@router.get("/info")
async def get_info():
    """Model info"""
    return {
        "project": "YOLO Test 1",
        "description": "Original YOLOv8PoseEstimator with tiling support",
        "status": "active",
        "available_sizes": ["n", "s", "m", "l", "x"],
        "loaded_models": list(pose_estimators.keys()),
        "features": [
            "Pose estimation",
            "Keypoint detection",
            "Image tiling for far-view scenes",
            "Video processing"
        ],
        "tile_grid": list(Config.TILE_GRID),
        "supports": ["images", "videos"]
    }

@router.post("/reload")
async def reload_model(model_size: str = Form('m')):
    """Reload specific model"""
    global pose_estimators
    
    if model_size in pose_estimators:
        del pose_estimators[model_size]
    
    estimator = get_estimator(model_size)
    
    return {
        "success": estimator is not None,
        "message": f"Model {model_size} reloaded" if estimator else "Failed to reload"
    }