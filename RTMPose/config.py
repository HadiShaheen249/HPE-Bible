"""
Configuration file for RTMPose + ByteTrack Pose Estimator
"""

from pathlib import Path
import os

class Config:
    """Project configuration"""
    
    # Base directory
    BASE_DIR = Path(__file__).parent.absolute()
    
    # ============================================
    # RTMPose Model Settings
    # ============================================
    # Available models:
    # - rtmpose-t (tiny)
    # - rtmpose-s (small) 
    # - rtmpose-m (medium) ← نستخدم ده
    # - rtmpose-l (large)
    
    POSE_MODEL_NAME = 'rtmpose-m'
    POSE_CONFIG = 'rtmpose-m_8xb256-420e_coco-256x192.py'
    POSE_CHECKPOINT = 'rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
    
    # ============================================
    # Detection Model Settings (for person detection)
    # ============================================
    # نستخدم RTMDet لكشف الأشخاص
    DET_MODEL_NAME = 'rtmdet-m'
    DET_CONFIG = 'rtmdet_m_8xb32-300e_coco.py'
    DET_CHECKPOINT = 'rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
    
    # ============================================
    # ByteTrack Settings
    # ============================================
    TRACK_THRESH = 0.5          # High threshold for first association
    TRACK_BUFFER = 30           # Number of frames to keep lost tracks
    MATCH_THRESH = 0.8          # IOU threshold for matching
    MIN_BOX_AREA = 10           # Minimum box area
    MOT20 = False               # Whether to use MOT20 settings
    FRAME_RATE = 30             # Assumed frame rate for tracking
    
    # ============================================
    # Confidence Thresholds
    # ============================================
    DET_CONFIDENCE = 0.5        # Detection confidence
    POSE_CONFIDENCE = 0.5       # Keypoint confidence
    
    # ============================================
    # Main Directories
    # ============================================
    MODELS_DIR = BASE_DIR / 'models'
    CONFIGS_DIR = BASE_DIR / 'configs'
    INPUT_DIR = BASE_DIR / 'input'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    # Model subdirectories
    DET_MODELS_DIR = MODELS_DIR / 'detection'
    POSE_MODELS_DIR = MODELS_DIR / 'pose'
    
    # Config subdirectories
    DET_CONFIGS_DIR = CONFIGS_DIR / 'detection'
    POSE_CONFIGS_DIR = CONFIGS_DIR / 'pose'
    
    # Input subdirectories
    INPUT_IMAGES_DIR = INPUT_DIR / 'images'
    INPUT_VIDEOS_DIR = INPUT_DIR / 'videos'
    
    # Output subdirectories
    OUTPUT_IMAGES_DIR = OUTPUT_DIR / 'images'
    OUTPUT_VIDEOS_DIR = OUTPUT_DIR / 'videos'
    OUTPUT_JSON_DIR = OUTPUT_DIR / 'json'
    OUTPUT_CSV_DIR = OUTPUT_DIR / 'csv'
    
    # ============================================
    # Display Settings
    # ============================================
    SHOW_LIVE = True
    SAVE_RESULTS = True
    
    # Visualization settings
    SKELETON_COLOR = (0, 255, 0)        # Green
    KEYPOINT_COLOR = (0, 0, 255)        # Red
    BBOX_COLOR = (255, 0, 0)            # Blue
    TRACK_ID_COLOR = (255, 255, 0)      # Yellow
    
    KEYPOINT_RADIUS = 5
    SKELETON_THICKNESS = 2
    BBOX_THICKNESS = 2
    TEXT_THICKNESS = 2
    TEXT_SCALE = 0.8
    
    # ============================================
    # COCO Keypoint Definitions (17 keypoints)
    # ============================================
    KEYPOINT_NAMES = [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'left_shoulder',  # 5
        'right_shoulder', # 6
        'left_elbow',     # 7
        'right_elbow',    # 8
        'left_wrist',     # 9
        'right_wrist',    # 10
        'left_hip',       # 11
        'right_hip',      # 12
        'left_knee',      # 13
        'right_knee',     # 14
        'left_ankle',     # 15
        'right_ankle'     # 16
    ]
    
    # ============================================
    # ✅ FIXED: Skeleton connections (COCO format - 0-indexed)
    # ============================================
    SKELETON_LINKS = [
        # Legs
        [15, 13], [13, 11],  # left leg: ankle->knee->hip
        [16, 14], [14, 12],  # right leg: ankle->knee->hip
        [11, 12],            # hips connection
        
        # Torso
        [5, 11],   # left shoulder -> left hip
        [6, 12],   # right shoulder -> right hip
        [5, 6],    # shoulders connection
        
        # Arms
        [5, 7],    # left shoulder -> left elbow
        [7, 9],    # left elbow -> left wrist
        [6, 8],    # right shoulder -> right elbow
        [8, 10],   # right elbow -> right wrist
        
        # Head
        [0, 1],    # nose -> left eye
        [0, 2],    # nose -> right eye
        [1, 3],    # left eye -> left ear
        [2, 4],    # right eye -> right ear
        [3, 5],    # left ear -> left shoulder
        [4, 6]     # right ear -> right shoulder
    ]
    
    # Colors for each keypoint (RGB)
    KEYPOINT_COLORS = [
        (255, 0, 0),     # nose - red
        (255, 85, 0),    # left_eye - orange
        (255, 170, 0),   # right_eye - yellow-orange
        (255, 255, 0),   # left_ear - yellow
        (170, 255, 0),   # right_ear - yellow-green
        (85, 255, 0),    # left_shoulder - green
        (0, 255, 0),     # right_shoulder - green
        (0, 255, 85),    # left_elbow - cyan-green
        (0, 255, 170),   # right_elbow - cyan
        (0, 255, 255),   # left_wrist - cyan
        (0, 170, 255),   # right_wrist - light blue
        (0, 85, 255),    # left_hip - blue
        (0, 0, 255),     # right_hip - blue
        (85, 0, 255),    # left_knee - purple
        (170, 0, 255),   # right_knee - purple
        (255, 0, 255),   # left_ankle - magenta
        (255, 0, 170)    # right_ankle - pink
    ]
    
    # ============================================
    # Device Settings
    # ============================================
    DEVICE = 'cuda:0'  # or 'cpu' if no GPU
    
    # ============================================
    # Performance Settings
    # ============================================
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    
    # ============================================
    # Model URLs (للتحميل التلقائي)
    # ============================================
    MODEL_URLS = {
        'rtmpose-m': {
            'config': 'https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py',
            'checkpoint': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        },
        'rtmdet-m': {
            'config': 'https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py',
            'checkpoint': 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
        }
    }
    
    # ============================================
    # Methods
    # ============================================
    
    @classmethod
    def get_det_model_path(cls) -> tuple:
        """
        Get detection model config and checkpoint paths
        
        Returns:
            tuple: (config_path, checkpoint_path)
        """
        config_path = cls.DET_CONFIGS_DIR / cls.DET_CONFIG
        checkpoint_path = cls.DET_MODELS_DIR / cls.DET_CHECKPOINT
        return config_path, checkpoint_path
    
    @classmethod
    def get_pose_model_path(cls) -> tuple:
        """
        Get pose model config and checkpoint paths
        
        Returns:
            tuple: (config_path, checkpoint_path)
        """
        config_path = cls.POSE_CONFIGS_DIR / cls.POSE_CONFIG
        checkpoint_path = cls.POSE_MODELS_DIR / cls.POSE_CHECKPOINT
        return config_path, checkpoint_path
    
    @classmethod
    def create_directories(cls):
        """Create all required directories"""
        directories = [
            cls.MODELS_DIR,
            cls.DET_MODELS_DIR,
            cls.POSE_MODELS_DIR,
            cls.CONFIGS_DIR,
            cls.DET_CONFIGS_DIR,
            cls.POSE_CONFIGS_DIR,
            cls.INPUT_IMAGES_DIR,
            cls.INPUT_VIDEOS_DIR,
            cls.OUTPUT_IMAGES_DIR,
            cls.OUTPUT_VIDEOS_DIR,
            cls.OUTPUT_JSON_DIR,
            cls.OUTPUT_CSV_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ All directories created successfully")
        print(f"📁 Detection models: {cls.DET_MODELS_DIR}")
        print(f"📁 Pose models: {cls.POSE_MODELS_DIR}")
        print(f"📁 Output images: {cls.OUTPUT_IMAGES_DIR}")
        print(f"📁 Output videos: {cls.OUTPUT_VIDEOS_DIR}")
    
    @classmethod
    def print_paths(cls):
        """Print all important paths"""
        print("\n" + "="*60)
        print("📂 Project Paths:")
        print("="*60)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"\n🔍 Detection Model:")
        print(f"  Config: {cls.DET_CONFIGS_DIR}")
        print(f"  Checkpoint: {cls.DET_MODELS_DIR}")
        print(f"\n🤸 Pose Model:")
        print(f"  Config: {cls.POSE_CONFIGS_DIR}")
        print(f"  Checkpoint: {cls.POSE_MODELS_DIR}")
        print(f"\n📥 Input:")
        print(f"  Images: {cls.INPUT_IMAGES_DIR}")
        print(f"  Videos: {cls.INPUT_VIDEOS_DIR}")
        print(f"\n📤 Output:")
        print(f"  Images: {cls.OUTPUT_IMAGES_DIR}")
        print(f"  Videos: {cls.OUTPUT_VIDEOS_DIR}")
        print(f"  JSON: {cls.OUTPUT_JSON_DIR}")
        print(f"  CSV: {cls.OUTPUT_CSV_DIR}")
        print("="*60 + "\n")
    
    @classmethod
    def print_model_info(cls):
        """Print model information"""
        print("\n" + "="*60)
        print("🤖 Model Configuration:")
        print("="*60)
        print(f"Detection Model: {cls.DET_MODEL_NAME}")
        print(f"Pose Model: {cls.POSE_MODEL_NAME}")
        print(f"Detection Confidence: {cls.DET_CONFIDENCE}")
        print(f"Pose Confidence: {cls.POSE_CONFIDENCE}")
        print(f"\n🎯 ByteTrack Settings:")
        print(f"  Track Threshold: {cls.TRACK_THRESH}")
        print(f"  Track Buffer: {cls.TRACK_BUFFER}")
        print(f"  Match Threshold: {cls.MATCH_THRESH}")
        print(f"  Min Box Area: {cls.MIN_BOX_AREA}")
        print(f"\n💻 Device: {cls.DEVICE}")
        print("="*60 + "\n")
    
    @classmethod
    def validate_device(cls):
        """Validate and set device"""
        import torch
        if cls.DEVICE.startswith('cuda') and not torch.cuda.is_available():
            print("⚠️  CUDA not available, switching to CPU")
            cls.DEVICE = 'cpu'
        return cls.DEVICE

# Create directories when importing
Config.create_directories()