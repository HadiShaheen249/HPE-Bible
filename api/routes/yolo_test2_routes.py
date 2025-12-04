"""
YOLO Test 2 API Routes - Uses Original Models with PlayerDetector & PoseEstimator
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add yolo_test2 to path
BASE_DIR = Path(__file__).parent.parent.parent
YOLO_TEST2_DIR = BASE_DIR / "yolo_test2"

if str(YOLO_TEST2_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_TEST2_DIR))

router = APIRouter()

# Global storage for models
models_cache = {}

def initialize_yolo_test2_models(det_size='m', pose_size='m'):
    """Initialize YOLO Test2 models with specific sizes"""
    global models_cache
    
    cache_key = f"det_{det_size}_pose_{pose_size}"
    
    if cache_key in models_cache:
        print(f"♻️  Using cached models: det={det_size}, pose={pose_size}")
        return models_cache[cache_key]
    
    try:
        # Change to yolo_test2 directory
        original_cwd = os.getcwd()
        os.chdir(YOLO_TEST2_DIR)
        
        print(f"🔄 Initializing YOLO Test2: Detection={det_size}, Pose={pose_size}")
        
        # ✅ استورد الـ Config أولاً
        from yolo_test2.config import Config
        
        # ✅ عدل الـ config
        Config.update_model_size(det_size)  # للـ detection
        
        # ✅ عدل الـ pose config يدوياً
        Config.POSE_CONFIG['model_name'] = f"yolov8{pose_size}-pose.pt"
        Config.POSE_CONFIG['model_path'] = Config.MODELS_DIR / f"yolov8{pose_size}-pose.pt"
        
        # ✅ دلوقتي استورد الموديلات (هتاخد الـ config المحدث)
        import importlib
        import yolo_test2.models as models_module
        importlib.reload(models_module)  # أعد تحميل الموديل عشان ياخد الـ config الجديد
        
        from yolo_test2.detector import PlayerDetector
        from yolo_test2.pose_estimator import PoseEstimator
        
        # Initialize models with updated config
        detector_model, pose_model, manager = models_module.initialize_models()
        
        # Create wrappers
        player_detector = PlayerDetector(detector_model)
        pose_estimator = PoseEstimator(pose_model)
        
        # Return to original directory
        os.chdir(original_cwd)
        
        # Cache the models
        models_cache[cache_key] = {
            'player_detector': player_detector,
            'pose_estimator': pose_estimator,
            'manager': manager,
            'det_size': det_size,
            'pose_size': pose_size
        }
        
        print(f"✅ YOLO Test2 initialized successfully: det={det_size}, pose={pose_size}")
        print(f"🔍 Detection model: yolov8{det_size}.pt")
        print(f"🧍 Pose model: yolov8{pose_size}-pose.pt")
        
        return models_cache[cache_key]
        
    except Exception as e:
        print(f"❌ Error initializing YOLO Test2: {e}")
        import traceback
        traceback.print_exc()
        try:
            os.chdir(original_cwd)
        except:
            pass
        return None

@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    mode: str = Form('pose'),
    pose_size: str = Form('m'),
    det_size: str = Form('m')
):
    """
    Detection or Pose using original YOLO Test2 classes
    
    Args:
        file: Image or video file
        mode: 'pose' or 'detection'
        pose_size: n, s, m, l, x (for pose estimation)
        det_size: n, s, m, l, x (for object detection)
    
    Returns:
        JSON response with results and download URL
    """
    
    # Initialize with appropriate sizes based on mode
    if mode == 'pose':
        models = initialize_yolo_test2_models(det_size='m', pose_size=pose_size)
    else:
        models = initialize_yolo_test2_models(det_size=det_size, pose_size='m')
    
    if models is None:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize YOLO Test2 models. Check server logs."
        )
    
    player_detector = models['player_detector']
    pose_estimator = models['pose_estimator']
    
    try:
        # Prepare directories
        input_dir = YOLO_TEST2_DIR / "input"
        output_dir = YOLO_TEST2_DIR / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        input_path = input_dir / filename
        
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Check if video or image
        file_ext = input_path.suffix.lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv']
        
        if is_video:
            # Process video
            print(f"🎬 Processing video with YOLO Test2 ({mode} mode)...")
            
            output_filename = f"result_{timestamp}.mp4"
            output_path = output_dir / output_filename
            
            cap = cv2.VideoCapture(str(input_path))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frame_count = 0
            poses_total = 0
            detections_total = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if mode == 'pose':
                    # Detect players first, then estimate poses
                    detections = player_detector.detect_players(frame)
                    result_frame = frame.copy()
                    
                    for det in detections:
                        # Crop player region
                        cropped = player_detector.crop_player(frame, det['bbox'])
                        if cropped is not None:
                            # Estimate pose on cropped image
                            pose_data = pose_estimator.estimate_pose(cropped)
                            if pose_data:
                                poses_total += 1
                                # Adjust keypoints to original frame coordinates
                                x1, y1, _, _ = det['bbox']
                                adjusted_pose = pose_data.copy()
                                adjusted_kpts = np.array(pose_data['keypoints'])
                                adjusted_kpts[:, 0] += x1
                                adjusted_kpts[:, 1] += y1
                                adjusted_pose['keypoints'] = adjusted_kpts.tolist()
                                
                                # Draw pose on frame
                                result_frame = pose_estimator.draw_pose(result_frame, adjusted_pose)
                    
                    # Draw detection boxes
                    result_frame = player_detector.draw_detections(result_frame, detections, draw_labels=False)
                    detections_total += len(detections)
                    
                else:
                    # Detection only mode
                    detections = player_detector.detect_players(frame)
                    result_frame = player_detector.draw_detections(frame, detections)
                    detections_total += len(detections)
                
                out.write(result_frame)
                frame_count += 1
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"   Processed {frame_count} frames...")
            
            cap.release()
            out.release()
            
            print(f"✅ Video processed: {frame_count} frames")
            
            return {
                "success": True,
                "message": f"Video {mode} completed",
                "type": "video",
                "mode": mode,
                "model_size": pose_size if mode == 'pose' else det_size,
                "frames_processed": frame_count,
                "total_detections": detections_total,
                "total_poses": poses_total if mode == 'pose' else 0,
                "download_url": f"/api/yolo-test2/result/{output_filename}"
            }
        
        else:
            # Process image
            print(f"📸 Processing image with YOLO Test2 ({mode} mode)...")
            
            img = cv2.imread(str(input_path))
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file")
            
            response_data = {
                "success": True,
                "message": f"{mode.capitalize()} completed",
                "type": "image",
                "mode": mode,
                "model_size": pose_size if mode == 'pose' else det_size
            }
            
            if mode == 'pose':
                # Pose estimation mode
                detections = player_detector.detect_players(img)
                result_img = img.copy()
                
                poses_found = 0
                for det in detections:
                    cropped = player_detector.crop_player(img, det['bbox'])
                    if cropped is not None:
                        pose_data = pose_estimator.estimate_pose(cropped)
                        if pose_data:
                            poses_found += 1
                            # Adjust keypoints to original image coordinates
                            x1, y1, _, _ = det['bbox']
                            adjusted_pose = pose_data.copy()
                            adjusted_kpts = np.array(pose_data['keypoints'])
                            adjusted_kpts[:, 0] += x1
                            adjusted_kpts[:, 1] += y1
                            adjusted_pose['keypoints'] = adjusted_kpts.tolist()
                            
                            result_img = pose_estimator.draw_pose(result_img, adjusted_pose)
                
                # Draw detection boxes
                result_img = player_detector.draw_detections(result_img, detections, draw_labels=False)
                
                response_data["num_persons"] = len(detections)
                response_data["poses_estimated"] = poses_found
                
                print(f"   Detected {len(detections)} persons, estimated {poses_found} poses")
            
            else:
                # Detection only mode
                detections = player_detector.detect_players(img)
                result_img = player_detector.draw_detections(img, detections)
                
                response_data["detections"] = [
                    {
                        "class_name": det['class_name'],
                        "confidence": float(det['confidence']),
                        "bbox": det['bbox']
                    } for det in detections
                ]
                response_data["num_detections"] = len(detections)
                
                print(f"   Detected {len(detections)} objects")
            
            # Save result image
            output_filename = f"result_{filename}"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), result_img)
            
            response_data["download_url"] = f"/api/yolo-test2/result/{output_filename}"
            
            print(f"✅ Image processed successfully")
            
            return response_data
    
    except Exception as e:
        import traceback
        error_detail = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"❌ Error processing file: {error_detail}")
        raise HTTPException(status_code=500, detail=error_detail)

@router.get("/result/{filename}")
async def get_result(filename: str):
    """
    Download result file (image or video)
    
    Args:
        filename: Name of the result file
    
    Returns:
        File response with the result
    """
    file_path = YOLO_TEST2_DIR / "output" / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"File not found: {filename}"
        )
    
    # Determine media type based on extension
    ext = file_path.suffix.lower()
    media_type = "video/mp4" if ext in ['.mp4', '.avi'] else "image/jpeg"
    
    return FileResponse(
        str(file_path), 
        media_type=media_type, 
        filename=filename
    )

@router.get("/info")
async def get_info():
    """
    Get information about YOLO Test 2 API
    
    Returns:
        JSON with project information
    """
    return {
        "project": "YOLO Test 2",
        "description": "Original PlayerDetector & PoseEstimator classes from yolo_test2 project",
        "status": "active",
        "available_sizes": ["n", "s", "m", "l", "x"],
        "loaded_models": list(models_cache.keys()),
        "modes": {
            "pose": "Detect players and estimate their poses",
            "detection": "Detect objects/players only"
        },
        "features": [
            "Object detection using YOLO",
            "Pose estimation with keypoints",
            "Player detection and cropping",
            "Multi-person tracking support",
            "Video processing support",
            "Configurable model sizes"
        ],
        "classes": {
            "PlayerDetector": "Detects and tracks players in images/videos",
            "PoseEstimator": "Estimates human pose with 17 COCO keypoints"
        },
        "supports": ["images (jpg, png)", "videos (mp4, avi, mov, mkv)"],
        "endpoints": {
            "detect": "POST /api/yolo-test2/detect",
            "result": "GET /api/yolo-test2/result/{filename}",
            "info": "GET /api/yolo-test2/info"
        }
    }

@router.post("/clear-cache")
async def clear_cache():
    """
    Clear models cache (useful for freeing memory)
    
    Returns:
        Success message
    """
    global models_cache
    
    cache_count = len(models_cache)
    models_cache.clear()
    
    return {
        "success": True,
        "message": f"Cleared {cache_count} cached model(s)",
        "cache_size": len(models_cache)
    }