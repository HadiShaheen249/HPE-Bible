"""
HPE Bible - Global Configuration
Central configuration for all projects
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

# Project Root Directory
ROOT_DIR = Path(__file__).parent.absolute()

# Project Directories
RTMPOSE_DIR = ROOT_DIR / "RTMPose"
YOLO_TEST1_DIR = ROOT_DIR / "yolo_test1"
YOLO_TEST2_DIR = ROOT_DIR / "yolo_test2"

# Model Directories
RTMPOSE_MODELS = RTMPOSE_DIR / "models"
YOLO_TEST1_MODELS = YOLO_TEST1_DIR / "models"
YOLO_TEST2_MODELS = YOLO_TEST2_DIR / "models"

# Input/Output Directories
RTMPOSE_INPUT = RTMPOSE_DIR / "input"
RTMPOSE_OUTPUT = RTMPOSE_DIR / "output"

YOLO_TEST1_INPUT = YOLO_TEST1_DIR / "input"
YOLO_TEST1_OUTPUT = YOLO_TEST1_DIR / "output"

YOLO_TEST2_INPUT = YOLO_TEST2_DIR / "input"
YOLO_TEST2_OUTPUT = YOLO_TEST2_DIR / "output"
YOLO_TEST2_LOGS = YOLO_TEST2_DIR / "logs"


class Settings(BaseSettings):
    """
    Global Settings for HPE Bible API
    Can be overridden with environment variables
    """
    
    # API Settings
    API_TITLE: str = "HPE Bible API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Unified API for Human Pose Estimation"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    RELOAD: bool = True
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".bmp"]
    ALLOWED_VIDEO_EXTENSIONS: list = [".mp4", ".avi", ".mov", ".mkv"]
    
    # Model Settings
    DEVICE: str = "cuda"  # or "cpu"
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    # RTMPose Settings
    RTMPOSE_ENABLED: bool = True
    RTMPOSE_MODEL_PATH: Optional[str] = None
    
    # YOLO Test1 Settings
    YOLO_TEST1_ENABLED: bool = True
    YOLO_TEST1_MODEL_NAME: str = "yolov8n-pose.pt"
    
    # YOLO Test2 Settings
    YOLO_TEST2_ENABLED: bool = True
    YOLO_TEST2_MODEL_NAME: str = "yolov8m.pt"
    YOLO_TEST2_POSE_MODEL: str = "yolov8m-pose.pt"
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache Settings
    ENABLE_CACHE: bool = False
    CACHE_TTL: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()


def ensure_directories():
    """
    Ensure all required directories exist
    """
    directories = [
        # RTMPose
        RTMPOSE_DIR,
        RTMPOSE_MODELS,
        RTMPOSE_INPUT,
        RTMPOSE_OUTPUT,
        
        # YOLO Test1
        YOLO_TEST1_DIR,
        YOLO_TEST1_MODELS,
        YOLO_TEST1_INPUT,
        YOLO_TEST1_OUTPUT,
        
        # YOLO Test2
        YOLO_TEST2_DIR,
        YOLO_TEST2_MODELS,
        YOLO_TEST2_INPUT,
        YOLO_TEST2_OUTPUT,
        YOLO_TEST2_LOGS,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory.name}/")


def get_model_path(project: str, model_name: str) -> Path:
    """
    Get full path for a model file
    
    Args:
        project: Project name (rtmpose, yolo_test1, yolo_test2)
        model_name: Model filename
    
    Returns:
        Full path to model file
    """
    project_models = {
        "rtmpose": RTMPOSE_MODELS,
        "yolo_test1": YOLO_TEST1_MODELS,
        "yolo_test2": YOLO_TEST2_MODELS,
    }
    
    if project not in project_models:
        raise ValueError(f"Unknown project: {project}")
    
    model_path = project_models[project] / model_name
    
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
    
    return model_path


def check_models():
    """
    Check if required models exist
    """
    print("\n" + "=" * 60)
    print("🔍 Checking Models...")
    print("=" * 60)
    
    models_to_check = {
        "RTMPose": [
            (RTMPOSE_MODELS, "rtmpose model files"),
        ],
        "YOLO Test1": [
            (YOLO_TEST1_MODELS / "yolov8n-pose.pt", "YOLOv8n-pose"),
            (YOLO_TEST1_MODELS / "yolov8x-pose.pt", "YOLOv8x-pose"),
        ],
        "YOLO Test2": [
            (YOLO_TEST2_MODELS / "yolov8m.pt", "YOLOv8m"),
            (YOLO_TEST2_MODELS / "yolov8m-pose.pt", "YOLOv8m-pose"),
            (YOLO_TEST2_MODELS / "yolov8x.pt", "YOLOv8x"),
            (YOLO_TEST2_MODELS / "yolov8x-pose.pt", "YOLOv8x-pose"),
        ],
    }
    
    for project, models in models_to_check.items():
        print(f"\n{project}:")
        for model_path, model_name in models:
            if isinstance(model_path, Path) and model_path.exists():
                if model_path.is_file():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
                    print(f"  ✅ {model_name}: {size_mb:.2f} MB")
                else:
                    # Check if directory has any .pt files
                    pt_files = list(model_path.glob("*.pt"))
                    if pt_files:
                        print(f"  ✅ {model_name}: {len(pt_files)} file(s)")
                    else:
                        print(f"  ⚠️  {model_name}: No .pt files found")
            else:
                print(f"  ❌ {model_name}: Not found")
    
    print("\n" + "=" * 60 + "\n")


def print_config():
    """
    Print current configuration
    """
    print("\n" + "=" * 60)
    print("⚙️  Configuration")
    print("=" * 60)
    print(f"API Title: {settings.API_TITLE}")
    print(f"Version: {settings.API_VERSION}")
    print(f"Host: {settings.HOST}:{settings.PORT}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Device: {settings.DEVICE}")
    print(f"Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
    print(f"Max Upload Size: {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f} MB")
    print("-" * 60)
    print(f"RTMPose: {'✅ Enabled' if settings.RTMPOSE_ENABLED else '❌ Disabled'}")
    print(f"YOLO Test1: {'✅ Enabled' if settings.YOLO_TEST1_ENABLED else '❌ Disabled'}")
    print(f"YOLO Test2: {'✅ Enabled' if settings.YOLO_TEST2_ENABLED else '❌ Disabled'}")
    print("=" * 60 + "\n")


# Run checks on import (optional)
if __name__ == "__main__":
    ensure_directories()
    print_config()
    check_models()