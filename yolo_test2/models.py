"""
Models loader and manager for Football Pose Estimation Project
Handles loading and initialization of YOLO models for detection and pose estimation
"""

import torch
import logging
from pathlib import Path
from ultralytics import YOLO
import sys

# Import configuration
try:
    from config import (
        DETECTION_CONFIG,
        POSE_CONFIG,
        MODELS_DIR,
        SELECTED_MODEL,
        LOGGING_CONFIG
    )
except ImportError:
    print("❌ Error: config.py file not found")
    print("Make sure config.py exists in the same directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"]),
    format=LOGGING_CONFIG["log_format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages YOLO models for object detection and pose estimation
    """
    
    def __init__(self):
        """Initialize Model Manager"""
        self.detection_model = None
        self.pose_model = None
        self.device = self._get_device()
        
        logger.info(f"🎮 Device used: {self.device}")
        logger.info(f"🤖 Selected model: YOLOv8{SELECTED_MODEL}")
        
    def _get_device(self):
        """
        Determine the best available device (CUDA, MPS, or CPU)
        
        Returns:
            str: Device name ('cuda', 'mps', or 'cpu')
        """
        device_config = DETECTION_CONFIG.get("device", "cuda").lower()
        
        # Check CUDA availability
        if device_config == "cuda" and torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU detected: {gpu_name}")
            
        # Check MPS (Apple Silicon) availability
        elif device_config == "mps" and torch.backends.mps.is_available():
            device = "mps"
            logger.info("✅ Apple Silicon (MPS) detected")
            
        # Fallback to CPU
        else:
            device = "cpu"
            if device_config != "cpu":
                logger.warning(f"⚠️ {device_config.upper()} not available, using CPU")
            else:
                logger.info("✅ CPU will be used")
                
        return device
    
    def load_detection_model(self):
        """
        Load YOLO object detection model
        
        Returns:
            YOLO: Loaded detection model
        """
        try:
            model_path = DETECTION_CONFIG["model_path"]
            model_name = DETECTION_CONFIG["model_name"]
            
            logger.info(f"📥 Loading Object Detection model: {model_name}")
            
            # Check if model exists locally
            if not model_path.exists():
                logger.info(f"📡 Model not found locally, downloading from the internet...")
                logger.info(f"💾 Model will be saved to: {model_path}")
                self.detection_model = YOLO(model_name)

                import shutil
                from pathlib import Path
                cache_path = Path.home() / '.cache' / 'ultralytics' / model_name
                if cache_path.exists():
                    shutil.copy(cache_path, model_path)
            else:
                logger.info(f"✅ Model found locally: {model_path}")
                self.detection_model = YOLO(str(model_path))
            
            self.detection_model.to(self.device)
            
            logger.info(f"✅ Object Detection model loaded successfully!")
            logger.info(f"📊 Number of classes: {len(self.detection_model.names)}")
            
            return self.detection_model
            
        except Exception as e:
            logger.error(f"❌ Error loading Object Detection model: {str(e)}")
            raise
    
    def load_pose_model(self):
        """
        Load YOLO pose estimation model
        
        Returns:
            YOLO: Loaded pose model
        """
        try:
            model_path = POSE_CONFIG["model_path"]
            model_name = POSE_CONFIG["model_name"]
            
            logger.info(f"📥 Loading Pose Estimation model: {model_name}")
            
            # Check if model exists locally
            if not model_path.exists():
                logger.info(f"📡 Model not found locally, downloading from the internet...")
                logger.info(f"💾 Model will be saved to: {model_path}")
                self.pose_model = YOLO(model_name)

                import shutil
                from pathlib import Path
                cache_path = Path.home() / '.cache' / 'ultralytics' / model_name
                if cache_path.exists():
                    shutil.copy(cache_path, model_path)
            else:
                logger.info(f"✅ Model found locally: {model_path}")
                self.pose_model = YOLO(str(model_path))
            
            self.pose_model.to(self.device)
            
            logger.info(f"✅ Pose Estimation model loaded successfully!")
            logger.info(f"🧍 Number of Keypoints: 17 (COCO format)")
            
            return self.pose_model
            
        except Exception as e:
            logger.error(f"❌ Error loading Pose Estimation model: {str(e)}")
            raise
    
    def load_all_models(self):
        """
        Load both detection and pose models
        
        Returns:
            tuple: (detection_model, pose_model)
        """
        logger.info("="*60)
        logger.info("🚀 Starting model loading")
        logger.info("="*60)
        
        detection_model = self.load_detection_model()
        pose_model = self.load_pose_model()
        
        logger.info("="*60)
        logger.info("✅ All models loaded successfully!")
        logger.info("="*60)
        
        return detection_model, pose_model
    
    def get_model_info(self):
        """
        Get information about loaded models
        
        Returns:
            dict: Model information
        """
        info = {
            "detection_model": {
                "name": DETECTION_CONFIG["model_name"],
                "path": str(DETECTION_CONFIG["model_path"]),
                "loaded": self.detection_model is not None,
                "device": self.device,
                "confidence_threshold": DETECTION_CONFIG["confidence_threshold"],
                "iou_threshold": DETECTION_CONFIG["iou_threshold"],
            },
            "pose_model": {
                "name": POSE_CONFIG["model_name"],
                "path": str(POSE_CONFIG["model_path"]),
                "loaded": self.pose_model is not None,
                "device": self.device,
                "confidence_threshold": POSE_CONFIG["confidence_threshold"],
                "num_keypoints": 17,
            },
            "selected_model_size": SELECTED_MODEL,
        }
        
        return info
    
    def print_model_info(self):
        """Print model information in a formatted way"""
        info = self.get_model_info()
        
        print("\n" + "="*60)
        print("📊 Model Information")
        print("="*60)
        
        print("\n🔍 Object Detection Model:")
        print(f"  - Name: {info['detection_model']['name']}")
        print(f"  - Path: {info['detection_model']['path']}")
        print(f"  - Loaded: {'✅ Yes' if info['detection_model']['loaded'] else '❌ No'}")
        print(f"  - Device: {info['detection_model']['device']}")
        print(f"  - Confidence: {info['detection_model']['confidence_threshold']}")
        print(f"  - IoU: {info['detection_model']['iou_threshold']}")
        
        print("\n🧍 Pose Estimation Model:")
        print(f"  - Name: {info['pose_model']['name']}")
        print(f"  - Path: {info['pose_model']['path']}")
        print(f"  - Loaded: {'✅ Yes' if info['pose_model']['loaded'] else '❌ No'}")
        print(f"  - Device: {info['pose_model']['device']}")
        print(f"  - Confidence: {info['pose_model']['confidence_threshold']}")
        print(f"  - Number of Keypoints: {info['pose_model']['num_keypoints']}")
        
        print(f"\n⭐ Selected Model Size: YOLOv8{info['selected_model_size'].upper()}")
        print("="*60 + "\n")


class DetectionModel:
    """Wrapper class for object detection"""
    
    def __init__(self, model, config):
        """
        Initialize detection model wrapper
        
        Args:
            model: YOLO model instance
            config: Detection configuration dict
        """
        self.model = model
        self.config = config
        
    def predict(self, image, conf=None, iou=None):
        """
        Run object detection on image
        
        Args:
            image: Input image (numpy array or path)
            conf: Confidence threshold (optional)
            iou: IoU threshold (optional)
            
        Returns:
            Results object from YOLO
        """
        conf = conf or self.config["confidence_threshold"]
        iou = iou or self.config["iou_threshold"]
        imgsz = self.config["imgsz"]
        
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=[self.config["target_class"]],
            verbose=False
        )
        
        return results[0] if results else None


class PoseModel:
    """Wrapper class for pose estimation"""
    
    def __init__(self, model, config):
        """
        Initialize pose model wrapper
        
        Args:
            model: YOLO pose model instance
            config: Pose configuration dict
        """
        self.model = model
        self.config = config
        
    def predict(self, image, conf=None):
        """
        Run pose estimation on image
        
        Args:
            image: Input image (numpy array or path)
            conf: Confidence threshold (optional)
            
        Returns:
            Results object from YOLO
        """
        conf = conf or self.config["confidence_threshold"]
        imgsz = self.config["imgsz"]
        
        results = self.model.predict(
            source=image,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )
        
        return results[0] if results else None


def initialize_models():
    """
    Initialize and load all models
    
    Returns:
        tuple: (DetectionModel, PoseModel, ModelManager)
    """
    try:
        manager = ModelManager()
        
        detection_model, pose_model = manager.load_all_models()
        
        detector = DetectionModel(detection_model, DETECTION_CONFIG)
        pose_estimator = PoseModel(pose_model, POSE_CONFIG)
        
        manager.print_model_info()
        
        return detector, pose_estimator, manager
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {str(e)}")
        raise


# Test function
if __name__ == "__main__":
    print("🧪 Testing model loading...\n")
    
    try:
        detector, pose_estimator, manager = initialize_models()
        print("\n✅ Models tested successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)
