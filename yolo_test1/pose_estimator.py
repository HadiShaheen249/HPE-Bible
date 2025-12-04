"""
YOLOv8 Pose Estimator Class
Enhanced version with image tiling for far-view scenes (e.g., soccer matches)
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple
import time
import os
from yolo_test1.config import Config


class YOLOv8PoseEstimator:
    """
    Professional class for pose estimation using YOLOv8
    """

    def __init__(self, model_name: str = 'yolov8n-pose.pt', conf_threshold: float = 0.5):
        """
        Initialize the model
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_path = None
        self._load_model()

    def _load_model(self):
        """Load model from models folder"""
        try:
            print(f"🔄 Loading model: {self.model_name}")
            self.model_path = Config.get_model_path(self.model_name)

            if self.model_path.exists():
                print(f"📦 Model found in: {self.model_path}")
                self.model = YOLO(str(self.model_path))
                print(f"✅ Model loaded successfully from models folder!")
            else:
                print(f"📥 Downloading model...")
                temp_model = YOLO(self.model_name)
                downloaded_path = self._find_downloaded_model()

                if downloaded_path and downloaded_path.exists():
                    import shutil
                    shutil.copy2(downloaded_path, self.model_path)
                    print(f"📦 Model copied to: {self.model_path}")
                    self.model = YOLO(str(self.model_path))
                    print(f"✅ Model loaded successfully!")
                else:
                    print(f"⚠️  Using model from default location")
                    self.model = temp_model

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _find_downloaded_model(self) -> Optional[Path]:
        """Find where ultralytics downloaded the model"""
        possible_locations = [
            Path.home() / '.ultralytics' / 'models' / self.model_name,
            Path(self.model_name),
            Path.cwd() / self.model_name,
        ]
        for location in possible_locations:
            if location.exists():
                return location
        return None

    def _draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """Draw keypoints and skeleton lines"""
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
            (255, 0, 170)
        ]
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        for connection in skeleton:
            start_idx = connection[0] - 1
            end_idx = connection[1] - 1
            if (start_idx < len(keypoints) and end_idx < len(keypoints)
                    and confidence[start_idx] > 0.5 and confidence[end_idx] > 0.5):
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        for idx, (keypoint, conf) in enumerate(zip(keypoints, confidence)):
            if conf > 0.5:
                x, y = int(keypoint[0]), int(keypoint[1])
                color = colors[idx % len(colors)]
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 2)
        return frame

    def _split_image_to_tiles(self, image, grid_size=(3, 3)):
        """
        تقسيم الصورة إلى مربعات صغيرة (tiles)
        Args:
            image: مصفوفة الصورة (numpy array)
            grid_size: (عدد الصفوف, عدد الأعمدة)
        Returns:
            قائمة تحتوي على tuples (crop, (x_offset, y_offset))
        """
        h, w, _ = image.shape
        rows, cols = grid_size
        tile_h, tile_w = h // rows, w // cols

        tiles = []
        for i in range(rows):
            for j in range(cols):
                x1, y1 = j * tile_w, i * tile_h
                x2, y2 = x1 + tile_w, y1 + tile_h
                crop = image[y1:y2, x1:x2]
                tiles.append((crop, (x1, y1)))
        return tiles

    # 🧩 Function to split frame into tiles (used for far-view frames)
    def _process_tiled_frame(self, frame: np.ndarray, grid_size=None) -> np.ndarray:
        """
        Process image or frame using tiling approach
        """
        # لو المستخدم غيّر حجم الـ grid من Config
        if grid_size is None:
            grid_size = Config.TILE_GRID

        tiles = self._split_image_to_tiles(frame, grid_size=grid_size)
        composed = frame.copy()

        for tile, (x_offset, y_offset) in tiles:
            results = self.model(tile, conf=self.conf_threshold, verbose=False)
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy()
                    for kpts, confs in zip(keypoints, confidences):
                        # نرجّع النقاط لمكانها الأصلي بالنسبة للصورة الكاملة
                        kpts[:, 0] += x_offset
                        kpts[:, 1] += y_offset
                        composed = self._draw_keypoints(composed, kpts, confs)
        return composed

    def predict_image(self, image_path: Union[str, Path],
                      save_result: bool = True,
                      output_path: Optional[str] = None) -> np.ndarray:
        """
        Analyze a single image — now with tiling support for distant scenes
        """
        print(f"📸 Processing image: {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # 👇 استخدم حجم grid من Config
        output_frame = self._process_tiled_frame(frame, grid_size=Config.TILE_GRID)

        if save_result:
            if output_path is None:
                input_filename = Path(image_path).name
                output_path = Config.OUTPUT_IMAGES_DIR / f"output_{input_filename}"
            else:
                output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), output_frame)
            print(f"💾 Result saved to: {output_path}")

        return output_frame

    def predict_video(self, video_path: Union[str, Path, int],
                      save_result: bool = True,
                      output_path: Optional[str] = None,
                      show_live: bool = True):
        """
        Analyze a video using tiling grid from Config
        """
        # Open video or camera
        is_camera = False
        if video_path == 0 or str(video_path).lower() == 'camera':
            cap = cv2.VideoCapture(0)
            is_camera = True
            print("📹 Opening camera...")
        else:
            cap = cv2.VideoCapture(str(video_path))
            print(f"📹 Processing video: {video_path}")

        if not cap.isOpened():
            raise ValueError("Cannot open video/camera")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        writer = None
        if save_result:
            if output_path is None:
                if is_camera:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = f"camera_output_{timestamp}.mp4"
                else:
                    input_filename = Path(video_path).name
                    output_filename = f"output_{input_filename}"
                output_path = Config.OUTPUT_VIDEOS_DIR / output_filename
            else:
                output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Saving video to: {output_path}")

        frame_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # 👇 استخدام grid الديناميكي من Config
                processed_frame = self._process_tiled_frame(frame, grid_size=Config.TILE_GRID)

                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(processed_frame, f'FPS: {current_fps:.2f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(processed_frame, f'Frames: {frame_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if writer is not None:
                    writer.write(processed_frame)

                if show_live:
                    cv2.imshow(f'YOLOv8 Pose Estimation Grid {Config.TILE_GRID} - Press Q to Exit', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("⏹️  Processing stopped by user")
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print("💾 Video saved successfully!")
            cv2.destroyAllWindows()
            elapsed = time.time() - start_time
            print(f"✅ Processed {frame_count} frames in {elapsed:.2f} seconds")
            print(f"📊 Average FPS: {frame_count / elapsed:.2f}")

def cleanup_default_models():
    """Optional: Clean up models from default ultralytics folder"""
    default_models_path = Path.home() / '.ultralytics' / 'models'
    if default_models_path.exists():
        print(f"\n🧹 Cleaning up default models folder...")
        for model_file in default_models_path.glob('*.pt'):
            try:
                model_file.unlink()
                print(f"   Deleted: {model_file.name}")
            except Exception as e:
                print(f"   Could not delete {model_file.name}: {e}")
        print("✅ Cleanup complete!")


# Quick test entry point
if __name__ == "__main__":
    Config.print_paths()
    estimator = YOLOv8PoseEstimator()
    print("✅ Ready for testing!")