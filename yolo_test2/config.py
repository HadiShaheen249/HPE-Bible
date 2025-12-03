"""
Configuration file for Football Pose Estimation Project
"""

import os
from pathlib import Path

# ==================== MODEL SELECTION ====================
def select_model():
    """
    Interactive model selection function
    """
    print("\n" + "="*60)
    print("🏆 Choosing the YOLOv8 model for Object Detection and Pose Estimation")
    print("="*60)
    
    print("\n📊 Model Comparison:\n")
    
    models_info = {
        'n': {
            'name': 'YOLOv8n (nano)',
            'speed': 'Very fast⚡⚡⚡⚡⚡ ',
            'accuracy': '⭐⭐⭐ Good',
            'size': '~6 MB',
            'best_for': 'CPU,Weak devices'
        },
        's': {
            'name': 'YOLOv8s (small)',
            'speed': '⚡⚡⚡⚡ fast ',
            'accuracy': '⭐⭐⭐⭐  Very good',
            'size': '~22 MB',
            'best_for': 'GPU Average, Real-time'
        },
        'm': {
            'name': 'YOLOv8m (medium) ⭐ Recommended',
            'speed': '⚡⚡⚡ Medium',
            'accuracy': '⭐⭐⭐⭐⭐ Excellent',
            'size': '~52 MB',
            'best_for': 'Good GPU, high accuracy'
        },
        'l': {
            'name': 'YOLOv8l (large)',
            'speed': '⚡⚡ Relatively slow',
            'accuracy': '⭐⭐⭐⭐⭐ Very excellent',
            'size': '~87 MB',
            'best_for': 'Strong GPU, maximum accuracy'
        },
        'x': {
            'name': 'YOLOv8x (xlarge)',
            'speed': '⚡ Slow',
            'accuracy': '⭐⭐⭐⭐⭐⭐ The best',
            'size': '~136 MB',
            'best_for': 'Professional GPU, research'
        }

    }
    
    for key, info in models_info.items():
        print(f"[{key.upper()}] {info['name']}")
        print(f"    Speed: {info['speed']}")
        print(f"    Accuracy: {info['accuracy']}")
        print(f"    Size: {info['size']}")
        print(f"    Best for: {info['best_for']}")
        print()
    
    print("="*60)
    
    while True:
        choice = input("\n🎯 Choose model (n/s/m/l/x) or press Enter for default [m]: ").strip().lower()
        
        if choice == '':
            choice = 'm'
            print(f"✅ Default model selected: YOLOv8m (medium)")
            break
        
        if choice in ['n', 's', 'm', 'l', 'x']:
            print(f"✅ Selected: {models_info[choice]['name']}")
            break
        else:
            print("❌ Invalid choice! Choose from (n/s/m/l/x)")

    print("="*60 + "\n")
    return choice


# Get user's model selection
SELECTED_MODEL = select_model()

# ==================== PROJECT PATHS ====================
# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Input/Output directories
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ==================== MODEL CONFIGURATION ====================

# Object Detection Model (YOLO)
DETECTION_CONFIG = {
    "model_name": f"yolov8{SELECTED_MODEL}.pt",
    "model_path": MODELS_DIR / f"yolov8{SELECTED_MODEL}.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_class": 0,  # 0 = person in COCO dataset
    "device": "cuda",  # Options: "cuda", "cpu", "mps" (for Mac M1/M2)
    "imgsz": 640,  # Image size for inference
}

# Pose Estimation Model
POSE_CONFIG = {
    "model_type": "yolov8",  # Options: "yolov8", "mediapipe"
    "model_name": f"yolov8{SELECTED_MODEL}-pose.pt",
    "model_path": MODELS_DIR / f"yolov8{SELECTED_MODEL}-pose.pt",
    "confidence_threshold": 0.5,
    "device": "cuda",
    "imgsz": 640,
}

# MediaPipe Configuration (if using MediaPipe)
MEDIAPIPE_CONFIG = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "model_complexity": 1,  # 0, 1, or 2 (higher = more accurate but slower)
    "enable_segmentation": False,
    "smooth_landmarks": True,
}

# ==================== VIDEO PROCESSING ====================
VIDEO_CONFIG = {
    "fps": 30,  # Output video FPS (None = use original)
    "frame_skip": 1,  # Process every Nth frame (1 = process all frames)
    "resize_width": None,  # Resize width (None = keep original)
    "resize_height": None,  # Resize height (None = keep original)
    "codec": "mp4v",  # Video codec: 'mp4v', 'XVID', 'H264'
    "save_frames": False,  # Save individual frames
}

# ==================== VISUALIZATION ====================
VISUALIZATION_CONFIG = {
    "draw_bboxes": True,  # Draw bounding boxes
    "draw_keypoints": True,  # Draw pose keypoints
    "draw_skeleton": True,  # Draw skeleton connections
    "draw_labels": True,  # Draw player IDs/labels
    
    # Colors (BGR format)
    "bbox_color": (0, 255, 0),  # Green
    "keypoint_color": (0, 0, 255),  # Red
    "skeleton_color": (255, 0, 0),  # Blue
    "text_color": (255, 255, 255),  # White
    
    # Sizes
    "bbox_thickness": 2,
    "keypoint_radius": 4,
    "skeleton_thickness": 2,
    "text_font": 0,  # cv2.FONT_HERSHEY_SIMPLEX
    "text_scale": 0.6,
    "text_thickness": 2,
}

# ==================== YOLO POSE KEYPOINTS ====================
# YOLO Pose keypoints indices (COCO format - 17 keypoints)
YOLO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Skeleton connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

# ==================== MEDIAPIPE POSE KEYPOINTS ====================
MEDIAPIPE_KEYPOINTS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

# ==================== DATA EXPORT ====================
EXPORT_CONFIG = {
    "save_json": True,  # Save keypoints as JSON
    "save_csv": True,  # Save keypoints as CSV
    "save_video": True,  # Save annotated video
    "save_images": False,  # Save annotated images
    "json_indent": 2,  # JSON formatting
}

# ==================== TRACKING CONFIGURATION ====================
TRACKING_CONFIG = {
    "enable_tracking": True,  # Enable player tracking across frames
    "max_disappeared": 30,  # Max frames before removing tracked object
    "max_distance": 50,  # Max pixel distance for tracking
}

# ==================== FILTERING ====================
FILTER_CONFIG = {
    "min_bbox_area": 1000,  # Minimum bounding box area (pixels²)
    "max_bbox_area": 500000,  # Maximum bounding box area (pixels²)
    "min_keypoints_visible": 5,  # Minimum visible keypoints for valid pose
}

# ==================== PERFORMANCE ====================
PERFORMANCE_CONFIG = {
    "use_half_precision": False,  # Use FP16 for faster inference (GPU only)
    "batch_size": 1,  # Batch processing (for multiple images)
    "num_workers": 4,  # DataLoader workers
}

# ==================== LOGGING ====================
LOGGING_CONFIG = {
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "log_file": LOGS_DIR / "pose_estimation.log",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# ==================== DEBUG MODE ====================
DEBUG = False  # Enable debug mode for verbose output

# ==================== PRINT CONFIGURATION SUMMARY ====================
def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*60)
    print("📋 Configuration Summary")
    print("="*60)
    print(f"🤖 Object Detection Model: yolov8{SELECTED_MODEL}.pt")
    print(f"🧍 Pose Estimation Model: yolov8{SELECTED_MODEL}-pose.pt")
    print(f"💾 Input Directory: {INPUT_DIR}")
    print(f"📤 Output Directory: {OUTPUT_DIR}")
    print(f"🎮 Device: {DETECTION_CONFIG['device']}")
    print(f"🎯 Confidence Threshold: {DETECTION_CONFIG['confidence_threshold']}")
    print("="*60 + "\n")

# Print configuration when module is loaded
if __name__ != "__main__":
    print_config_summary()
