"""
Utility functions for RTMPose + ByteTrack Pose Estimator
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Dict, Optional
import json
import time
from datetime import datetime

# ============================================
# ✅ FIXED: Import pandas at module level with error handling
# ============================================
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not installed, CSV export will be disabled")


class VideoInfo:
    """Video information class"""
    
    @staticmethod
    def get_video_info(video_path: Union[str, Path]) -> Optional[dict]:
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
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(int(cap.get(cv2.CAP_PROP_FPS)), 1),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
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
            print("="*50)
            print(f"📐 Dimensions: {info['width']}x{info['height']}")
            print(f"🎬 FPS: {info['fps']}")
            print(f"📊 Frame Count: {info['frame_count']}")
            print(f"⏱️  Duration: {info['duration']:.2f} seconds")
            print("="*50)
        else:
            print("❌ Could not read video information")


class ImageInfo:
    """Image information class"""
    
    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> Optional[dict]:
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
            'size': Path(image_path).stat().st_size / 1024,  # KB
            'dtype': str(img.dtype)
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
            print("="*50)
            print(f"📐 Dimensions: {info['width']}x{info['height']}")
            print(f"🎨 Channels: {info['channels']}")
            print(f"💾 Size: {info['size']:.2f} KB")
            print(f"📊 Data Type: {info['dtype']}")
            print("="*50)
        else:
            print("❌ Could not read image information")


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
        if not directory.exists():
            return []
        
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
        
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
        if not directory.exists():
            return []
        
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        
        videos = []
        for ext in extensions:
            videos.extend(directory.glob(f'*{ext}'))
            videos.extend(directory.glob(f'*{ext.upper()}'))
        
        return sorted(videos)
    
    @staticmethod
    def create_output_path(input_path: Union[str, Path], 
                          output_dir: Union[str, Path],
                          prefix: str = 'output_',
                          suffix: str = '') -> Path:
        """
        Create output path for processed file
        
        Args:
            input_path: Input file path
            output_dir: Output directory
            prefix: Prefix for output filename
            suffix: Suffix for output filename
            
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stem = input_path.stem
        ext = input_path.suffix
        
        return output_dir / f"{prefix}{stem}{suffix}{ext}"
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Get current timestamp string
        
        Returns:
            Timestamp string in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")


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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 JSON results saved to: {output_path}")
    
    @staticmethod
    def save_keypoints_csv(keypoints_data: List[dict], output_path: Union[str, Path]):
        """
        Save keypoints to CSV file
        
        Args:
            keypoints_data: List of keypoints dictionaries
            output_path: Output CSV file path
        """
        # ============================================
        # ✅ FIXED: Better pandas handling
        # ============================================
        if not PANDAS_AVAILABLE:
            print("⚠️  pandas not installed, skipping CSV export")
            print("💡 Install with: pip install pandas")
            return
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(keypoints_data)
            df.to_csv(output_path, index=False)
            
            print(f"💾 Keypoints CSV saved to: {output_path}")
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
    
    @staticmethod
    def save_tracking_results(tracking_data: List[dict], output_path: Union[str, Path]):
        """
        Save tracking results to JSON file
        
        Args:
            tracking_data: List of tracking dictionaries
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(tracking_data),
            'tracking_data': tracking_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Tracking results saved to: {output_path}")


class VisualizationHelper:
    """Visualization helper class"""
    
    @staticmethod
    def draw_bbox(frame: np.ndarray, 
                  bbox: Tuple[int, int, int, int],
                  track_id: Optional[int] = None,
                  color: Tuple[int, int, int] = (255, 0, 0),
                  thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on frame
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Track ID (optional)
            color: Box color
            thickness: Line thickness
            
        Returns:
            Frame with drawn bbox
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        if track_id is not None:
            label = f"ID: {track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, -1)
            
            # Draw text
            cv2.putText(frame, label,
                       (x1 + 5, y1 - 5),
                       font, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    @staticmethod
    def draw_keypoints(frame: np.ndarray,
                      keypoints: np.ndarray,
                      confidences: np.ndarray,
                      keypoint_colors: List[Tuple[int, int, int]],
                      radius: int = 5,
                      conf_threshold: float = 0.5) -> np.ndarray:
        """
        Draw keypoints on frame
        
        Args:
            frame: Input frame
            keypoints: Keypoint coordinates (N, 2)
            confidences: Keypoint confidences (N,)
            keypoint_colors: Colors for each keypoint
            radius: Circle radius
            conf_threshold: Confidence threshold
            
        Returns:
            Frame with drawn keypoints
        """
        for idx, (kpt, conf) in enumerate(zip(keypoints, confidences)):
            if conf > conf_threshold:
                x, y = int(kpt[0]), int(kpt[1])
                color = keypoint_colors[idx % len(keypoint_colors)]
                
                # Draw filled circle
                cv2.circle(frame, (x, y), radius, color, -1)
                # Draw border
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
        
        return frame
    
    @staticmethod
    def draw_skeleton(frame: np.ndarray,
                     keypoints: np.ndarray,
                     confidences: np.ndarray,
                     skeleton_links: List[List[int]],
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2,
                     conf_threshold: float = 0.5) -> np.ndarray:
        """
        Draw skeleton connections on frame
        
        Args:
            frame: Input frame
            keypoints: Keypoint coordinates (N, 2)
            confidences: Keypoint confidences (N,)
            skeleton_links: List of keypoint pairs to connect
            color: Line color
            thickness: Line thickness
            conf_threshold: Confidence threshold
            
        Returns:
            Frame with drawn skeleton
        """
        for link in skeleton_links:
            # ============================================
            # ✅ FIXED: Better index handling for skeleton links
            # ============================================
            start_idx = link[0]
            end_idx = link[1]
            
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                confidences[start_idx] > conf_threshold and 
                confidences[end_idx] > conf_threshold):
                
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                
                cv2.line(frame, start_point, end_point, color, thickness)
        
        return frame
    
    @staticmethod
    def put_fps_text(frame: np.ndarray, 
                    fps: float,
                    position: Tuple[int, int] = (10, 30),
                    color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Put FPS text on frame
        
        Args:
            frame: Input frame
            fps: FPS value
            position: Text position
            color: Text color
            
        Returns:
            Frame with FPS text
        """
        text = f'FPS: {fps:.2f}'
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame
    
    @staticmethod
    def put_info_text(frame: np.ndarray,
                     text: str,
                     position: Tuple[int, int] = (10, 70),
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Put info text on frame
        
        Args:
            frame: Input frame
            text: Text to display
            position: Text position
            color: Text color
            
        Returns:
            Frame with info text
        """
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame


class PerformanceMonitor:
    """Performance monitoring class"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.start_time = None
        self.frame_times = []
        self.frame_count = 0
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.frame_times = []
        self.frame_count = 0
    
    def add_frame(self):
        """Add frame timing"""
        if self.start_time:
            self.frame_times.append(time.time())
            self.frame_count += 1
    
    def get_current_fps(self) -> float:
        """
        Get current FPS
        
        Returns:
            Current FPS value
        """
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate FPS from last N frames
        n_frames = min(30, len(self.frame_times))
        time_diff = self.frame_times[-1] - self.frame_times[-n_frames]
        
        if time_diff > 0:
            return n_frames / time_diff
        return 0.0
    
    def get_stats(self) -> Optional[dict]:
        """
        Get performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.start_time or not self.frame_times:
            return None
        
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        return {
            'total_time': total_time,
            'frame_count': self.frame_count,
            'avg_fps': avg_fps,
            'current_fps': self.get_current_fps()
        }
    
    def print_stats(self):
        """Print performance statistics"""
        stats = self.get_stats()
        if stats:
            print("\n📊 Performance Statistics:")
            print("="*50)
            print(f"⏱️  Total Time: {stats['total_time']:.2f} seconds")
            print(f"📊 Frame Count: {stats['frame_count']}")
            print(f"📈 Average FPS: {stats['avg_fps']:.2f}")
            print(f"⚡ Current FPS: {stats['current_fps']:.2f}")
            print("="*50)


class ModelDownloader:
    """Model download helper class"""
    
    @staticmethod
    def download_file(url: str, output_path: Union[str, Path]) -> bool:
        """
        Download file from URL
        
        Args:
            url: Download URL
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import urllib.request
            
            # ============================================
            # ✅ FIXED: Better tqdm handling
            # ============================================
            try:
                from tqdm import tqdm
                TQDM_AVAILABLE = True
            except ImportError:
                TQDM_AVAILABLE = False
                print("💡 Tip: Install tqdm for progress bar: pip install tqdm")
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"📥 Downloading from: {url}")
            print(f"📁 Saving to: {output_path}")
            
            if TQDM_AVAILABLE:
                class DownloadProgressBar(tqdm):
                    def update_to(self, b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            self.total = tsize
                        self.update(b * bsize - self.n)
                
                with DownloadProgressBar(unit='B', unit_scale=True,
                                        miniters=1, desc=output_path.name) as t:
                    urllib.request.urlretrieve(url, filename=output_path,
                                             reporthook=t.update_to)
            else:
                # Download without progress bar
                urllib.request.urlretrieve(url, filename=output_path)
            
            print(f"✅ Download completed!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False


def test_utils():
    """Test utility functions"""
    print("🧪 Testing utility functions...")
    
    # Test FileManager
    print("\n📁 Testing FileManager:")
    images = FileManager.get_all_images('input/images')
    print(f"Number of images found: {len(images)}")
    
    videos = FileManager.get_all_videos('input/videos')
    print(f"Number of videos found: {len(videos)}")
    
    # Test timestamp
    print(f"\n⏰ Current timestamp: {FileManager.get_timestamp()}")
    
    # Test PerformanceMonitor
    print("\n⏱️  Testing PerformanceMonitor:")
    monitor = PerformanceMonitor()
    monitor.start()
    
    for i in range(10):
        time.sleep(0.1)
        monitor.add_frame()
    
    monitor.print_stats()
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_utils()