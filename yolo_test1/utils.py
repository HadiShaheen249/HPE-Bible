"""
Utility functions for YOLOv8 Pose Estimator
Utility functions
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union
import json


class VideoInfo:
    """Video information class"""
    
    @staticmethod
    def get_video_info(video_path: Union[str, Path]) -> dict:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video information
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / int(cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        return info
    
    @staticmethod
    def print_video_info(video_path: Union[str, Path]):
        """
        Print video information
        
        Args:
            video_path: Path to video file
        """
        info = VideoInfo.get_video_info(video_path)
        
        if info:
            print("\n📹 Video Information:")
            print("="*40)
            print(f"📐 Dimensions: {info['width']}x{info['height']}")
            print(f"🎬 FPS: {info['fps']}")
            print(f"📊 Frame Count: {info['frame_count']}")
            print(f"⏱️  Duration: {info['duration']:.2f} seconds")
            print("="*40)


class ImageInfo:
    """Image information class"""
    
    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """
        Get image information
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing image information
        """
        img = cv2.imread(str(image_path))
        
        if img is None:
            return None
        
        info = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': img.shape[2] if len(img.shape) > 2 else 1,
            'size': Path(image_path).stat().st_size / 1024  # KB
        }
        
        return info
    
    @staticmethod
    def print_image_info(image_path: Union[str, Path]):
        """
        Print image information
        
        Args:
            image_path: Path to image file
        """
        info = ImageInfo.get_image_info(image_path)
        
        if info:
            print("\n📸 Image Information:")
            print("="*40)
            print(f"📐 Dimensions: {info['width']}x{info['height']}")
            print(f"🎨 Channels: {info['channels']}")
            print(f"💾 Size: {info['size']:.2f} KB")
            print("="*40)


class FileManager:
    """File management class"""
    
    @staticmethod
    def get_all_images(directory: Union[str, Path]) -> List[Path]:
        """
        Get all images in directory
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of image file paths
        """
        directory = Path(directory)
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        images = []
        for ext in extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(images)
    
    @staticmethod
    def get_all_videos(directory: Union[str, Path]) -> List[Path]:
        """
        Get all videos in directory
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of video file paths
        """
        directory = Path(directory)
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        
        videos = []
        for ext in extensions:
            videos.extend(directory.glob(f'*{ext}'))
            videos.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(videos)
    
    @staticmethod
    def create_output_path(input_path: Union[str, Path], 
                          output_dir: Union[str, Path],
                          prefix: str = 'output_') -> Path:
        """
        Create output path for processed file
        
        Args:
            input_path: Input file path
            output_dir: Output directory
            prefix: Prefix for output filename
            
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir / f"{prefix}{input_path.name}"


class ResultSaver:
    """Results saving class"""
    
    @staticmethod
    def save_results_json(results: dict, output_path: Union[str, Path]):
        """
        Save results to JSON file
        
        Args:
            results: Results dictionary to save
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 JSON results saved to: {output_path}")
    
    @staticmethod
    def save_keypoints_csv(keypoints: np.ndarray, output_path: Union[str, Path]):
        """
        Save keypoints to CSV file
        
        Args:
            keypoints: Keypoints array
            output_path: Output CSV file path
        """
        import pandas as pd
        
        # Create DataFrame
        data = []
        for frame_idx, frame_kpts in enumerate(keypoints):
            for person_idx, person_kpts in enumerate(frame_kpts):
                for kpt_idx, (x, y, conf) in enumerate(person_kpts):
                    data.append({
                        'frame': frame_idx,
                        'person': person_idx,
                        'keypoint': kpt_idx,
                        'x': x,
                        'y': y,
                        'confidence': conf
                    })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        
        print(f"💾 Keypoints CSV saved to: {output_path}")


class PerformanceMonitor:
    """Performance monitoring class"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = None
        self.frame_times = []
    
    def start(self):
        """Start timing"""
        import time
        self.start_time = time.time()
        self.frame_times = []
    
    def add_frame(self):
        """Add frame timing"""
        import time
        if self.start_time:
            self.frame_times.append(time.time())
    
    def get_stats(self) -> dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        import time
        if not self.start_time or not self.frame_times:
            return None
        
        total_time = time.time() - self.start_time
        frame_count = len(self.frame_times)
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        return {
            'total_time': total_time,
            'frame_count': frame_count,
            'avg_fps': avg_fps
        }
    
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        if stats:
            print("\n📊 Performance Statistics:")
            print("="*40)
            print(f"⏱️  Total Time: {stats['total_time']:.2f} seconds")
            print(f"📊 Frame Count: {stats['frame_count']}")
            print(f"📈 Average FPS: {stats['avg_fps']:.2f}")
            print("="*40)


def test_utils():
    """Test utility functions"""
    print("🧪 Testing utility functions...")
    
    # Test FileManager
    print("\n📁 Found files:")
    images = FileManager.get_all_images('input/images')
    print(f"Number of images: {len(images)}")
    
    videos = FileManager.get_all_videos('input/videos')
    print(f"Number of videos: {len(videos)}")
    
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    test_utils()