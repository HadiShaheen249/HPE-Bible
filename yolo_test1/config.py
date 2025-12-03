"""
Configuration file for YOLOv8 Pose Estimator
Configuration file
"""

from pathlib import Path
import os


class Config:
    """Project configuration"""
    
    # Base directory - automatically detect project root
    BASE_DIR = Path(__file__).parent.absolute()
    
    # Model settings
    MODEL_NAME = 'yolov8n-pose.pt'  # n, s, m, l, x
    CONFIDENCE_THRESHOLD = 0.5
    
    # Main directories
    MODELS_DIR = BASE_DIR / 'models'
    INPUT_DIR = BASE_DIR / 'input'
    OUTPUT_DIR = BASE_DIR / 'output'
    
    # Input subdirectories
    INPUT_IMAGES_DIR = INPUT_DIR / 'images'
    INPUT_VIDEOS_DIR = INPUT_DIR / 'videos'
    
    # Output subdirectories
    OUTPUT_IMAGES_DIR = OUTPUT_DIR / 'images'
    OUTPUT_VIDEOS_DIR = OUTPUT_DIR / 'videos'
    
    # Display settings
    SHOW_LIVE = True
    SAVE_RESULTS = True

    # Default tiling grid for far-view analysis (you can adjust it)
    TILE_GRID = (3, 3)
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """
        Get full path for model file
        
        Args:
            model_name: Name of the model
            
        Returns:
            Full path to model file
        """
        return cls.MODELS_DIR / model_name
    
    @classmethod
    def create_directories(cls):
        """Create all required directories"""
        directories = [
            cls.MODELS_DIR,
            cls.INPUT_IMAGES_DIR,
            cls.INPUT_VIDEOS_DIR,
            cls.OUTPUT_IMAGES_DIR,
            cls.OUTPUT_VIDEOS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("✅ All directories created successfully")
        print(f"📁 Models directory: {cls.MODELS_DIR}")
        print(f"📁 Output images: {cls.OUTPUT_IMAGES_DIR}")
        print(f"📁 Output videos: {cls.OUTPUT_VIDEOS_DIR}")
    
    @classmethod
    def print_paths(cls):
        """Print all important paths"""
        print("\n" + "="*60)
        print("📂 Project Paths:")
        print("="*60)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Models: {cls.MODELS_DIR}")
        print(f"Input Images: {cls.INPUT_IMAGES_DIR}")
        print(f"Input Videos: {cls.INPUT_VIDEOS_DIR}")
        print(f"Output Images: {cls.OUTPUT_IMAGES_DIR}")
        print(f"Output Videos: {cls.OUTPUT_VIDEOS_DIR}")
        print("="*60 + "\n")


# Create directories when importing the file
Config.create_directories()