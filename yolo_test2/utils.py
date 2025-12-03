"""
Utility functions for Football Pose Estimation Project
Includes helper functions for image/video processing, data export, and visualization
"""

import cv2
import numpy as np
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import sys

# Import configuration
try:
    from config import (
        OUTPUT_DIR,
        EXPORT_CONFIG,
        VIDEO_CONFIG,
        VISUALIZATION_CONFIG,
        YOLO_KEYPOINTS,
        LOGGING_CONFIG
    )
except ImportError:
    print("❌ Error: config.py not found")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)


# ==================== VIDEO PROCESSING ====================

class VideoReader:
    """
    Video reader with frame extraction capabilities
    """
    
    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video reader
        
        Args:
            video_path: Path to video file
        """
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"📹 Video opened: {Path(video_path).name}")
        logger.info(f"   Dimensions: {self.width}x{self.height}")
        logger.info(f"   FPS: {self.fps}")
        logger.info(f"   Frame count: {self.frame_count}")
        logger.info(f"   Duration: {self.duration:.2f} seconds")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_frame_at_index(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame at specific index"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_frame_at_time(self, time_sec: float) -> Optional[np.ndarray]:
        """Get frame at specific timestamp"""
        frame_idx = int(time_sec * self.fps)
        return self.get_frame_at_index(frame_idx)
    
    def reset(self):
        """Reset video to beginning"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self):
        """Release video capture"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __len__(self):
        return self.frame_count


class VideoWriter:
    """
    Video writer with customizable output settings
    """
    
    def __init__(self, output_path: Union[str, Path], 
                 fps: int, width: int, height: int,
                 codec: str = 'mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Output video path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec
        """
        self.output_path = str(output_path)
        self.fps = fps
        self.width = width
        self.height = height
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, fps, (width, height)
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video file: {output_path}")
        
        self.frame_count = 0
        logger.info(f"📹 Video writer created: {Path(output_path).name}")
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to video"""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release writer"""
        self.writer.release()
        logger.info(f"✅ Video saved: {self.frame_count} frames")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ==================== IMAGE PROCESSING ====================

def resize_image(image: np.ndarray, 
                width: Optional[int] = None, 
                height: Optional[int] = None,
                keep_aspect_ratio: bool = True) -> np.ndarray:
    """Resize image with optional aspect ratio"""
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if keep_aspect_ratio:
        if width and not height:
            ratio = width / w
            height = int(h * ratio)
        elif height and not width:
            ratio = height / h
            width = int(w * ratio)
    else:
        width = width or w
        height = height or h
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def crop_image(image: np.ndarray, bbox: List[int], padding: int = 0) -> np.ndarray:
    """Crop image using bounding box"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def add_text_with_background(image: np.ndarray,
                             text: str,
                             position: Tuple[int, int],
                             font_scale: float = 0.6,
                             thickness: int = 2,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0),
                             padding: int = 5) -> np.ndarray:
    """Add text with a solid background to image"""
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    cv2.rectangle(output,
                 (x - padding, y - text_h - padding),
                 (x + text_w + padding, y + baseline + padding),
                 bg_color, -1)
    
    cv2.putText(output, text, (x, y),
               font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return output


def create_grid_layout(images: List[np.ndarray],
                      grid_size: Optional[Tuple[int, int]] = None,
                      cell_size: Tuple[int, int] = (300, 300),
                      padding: int = 10,
                      bg_color: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
    """Create a grid of images"""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_images = len(images)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    cell_w, cell_h = cell_size
    
    grid_h = rows * cell_h + (rows + 1) * padding
    grid_w = cols * cell_w + (cols + 1) * padding
    grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        
        img_resized = cv2.resize(img, (cell_w, cell_h))
        
        y = row * cell_h + (row + 1) * padding
        x = col * cell_w + (col + 1) * padding
        
        grid[y:y+cell_h, x:x+cell_w] = img_resized
    
    return grid


# ==================== DATA EXPORT ====================

def save_detections_json(detections: List[Dict],
                        frame_idx: int,
                        output_path: Union[str, Path],
                        metadata: Optional[Dict] = None):
    """Save detections JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'frame_index': frame_idx,
        'timestamp': datetime.now().isoformat(),
        'num_detections': len(detections),
        'detections': detections
    }
    
    if metadata:
        data['metadata'] = metadata
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=EXPORT_CONFIG['json_indent'])
    
    logger.debug(f"💾 JSON saved: {output_path}")


def save_poses_json(poses: List[Dict],
                   frame_idx: int,
                   output_path: Union[str, Path],
                   metadata: Optional[Dict] = None):
    """Save poses JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'frame_index': frame_idx,
        'timestamp': datetime.now().isoformat(),
        'num_poses': len([p for p in poses if p is not None]),
        'poses': poses,
        'keypoint_names': YOLO_KEYPOINTS
    }
    
    if metadata:
        data['metadata'] = metadata
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=EXPORT_CONFIG['json_indent'])
    
    logger.debug(f"💾 JSON saved: {output_path}")


def save_poses_csv(poses_data: List[Dict],
                  output_path: Union[str, Path]):
    """Save poses CSV"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ['frame_idx', 'player_id']
    
    for kpt_idx, kpt_name in YOLO_KEYPOINTS.items():
        headers.extend([f'{kpt_name}_x', f'{kpt_name}_y', f'{kpt_name}_conf'])
    
    headers.extend(['visible_keypoints', 'avg_confidence'])
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for data in poses_data:
            frame_idx = data['frame_idx']
            poses = data['poses']
            
            for player_id, pose in enumerate(poses):
                if pose is None:
                    continue
                
                row = {
                    'frame_idx': frame_idx,
                    'player_id': player_id
                }
                
                keypoints = pose['keypoints']
                for kpt_idx, kpt_name in YOLO_KEYPOINTS.items():
                    if kpt_idx < len(keypoints):
                        x, y, conf = keypoints[kpt_idx]
                        row[f'{kpt_name}_x'] = x
                        row[f'{kpt_name}_y'] = y
                        row[f'{kpt_name}_conf'] = conf
                    else:
                        row[f'{kpt_name}_x'] = 0
                        row[f'{kpt_name}_y'] = 0
                        row[f'{kpt_name}_conf'] = 0
                
                row['visible_keypoints'] = pose['visible_keypoints']
                row['avg_confidence'] = pose.get('confidence', 0)
                
                writer.writerow(row)
    
    logger.info(f"💾 CSV saved: {output_path}")


def load_poses_json(json_path: Union[str, Path]) -> Dict:
    """Load poses JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== PROGRESS TRACKING ====================

class ProgressTracker:
    """
    Simple progress tracker for processing
    """
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        percent = (self.current / self.total) * 100
        filled = int(percent / 2)
        bar = '█' * filled + '░' * (50 - filled)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
        
        print(f'\r{self.desc}: |{bar}| {percent:.1f}% '
              f'[{self.current}/{self.total}] '
              f'ETA: {eta:.1f}s', end='', flush=True)
        
        if self.current >= self.total:
            print()
    
    def close(self):
        print()


# ==================== FILE UTILITIES ====================

def get_output_filename(input_path: Union[str, Path],
                       suffix: str = "_output",
                       extension: Optional[str] = None) -> Path:
    """Generate output filename"""
    input_path = Path(input_path)
    
    if extension is None:
        extension = input_path.suffix
    
    output_name = f"{input_path.stem}{suffix}{extension}"
    return OUTPUT_DIR / output_name


def ensure_output_dir(subdir: Optional[str] = None) -> Path:
    """Ensure output directory exists"""
    if subdir:
        output_dir = OUTPUT_DIR / subdir
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_supported_image_extensions() -> List[str]:
    return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']


def get_supported_video_extensions() -> List[str]:
    return ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']


def is_image_file(file_path: Union[str, Path]) -> bool:
    return Path(file_path).suffix.lower() in get_supported_image_extensions()


def is_video_file(file_path: Union[str, Path]) -> bool:
    return Path(file_path).suffix.lower() in get_supported_video_extensions()


# Test function
if __name__ == "__main__":
    print("🧪 Utils Test...\n")
    
    print("✅ All utility functions loaded successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🖼️ Supported image formats: {get_supported_image_extensions()}")
    print(f"🎥 Supported video formats: {get_supported_video_extensions()}")
