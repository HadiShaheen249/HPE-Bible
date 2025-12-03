"""
Video Processor for Football Pose Estimation
Processes videos frame by frame with detection and pose estimation
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import sys

import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import locale
    locale.setlocale(locale.LC_ALL, '')

# Import configuration and modules
try:
    from config import (
        VIDEO_CONFIG,
        EXPORT_CONFIG,
        OUTPUT_DIR,
        TRACKING_CONFIG,
        VISUALIZATION_CONFIG
    )
    from utils import (
        VideoReader,
        VideoWriter,
        ProgressTracker,
        save_poses_json,
        save_poses_csv,
        get_output_filename,
        ensure_output_dir,
        is_video_file,
        add_text_with_background
    )
    from detector import PlayerDetector, MultiPlayerDetector
    from pose_estimator import PoseEstimator, PoseAnalyzer
except ImportError as e:
    print(f"❌ Import error: {str(e)}")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processor for football match pose estimation
    """
    
    def __init__(self, detector: PlayerDetector, 
                 pose_estimator: PoseEstimator,
                 enable_tracking: bool = True):
        """
        Initialize Video Processor
        
        Args:
            detector: PlayerDetector instance
            pose_estimator: PoseEstimator instance
            enable_tracking: Whether to enable player tracking
        """
        # Use MultiPlayerDetector if tracking is enabled
        if enable_tracking and TRACKING_CONFIG['enable_tracking']:
            self.detector = MultiPlayerDetector(detector.model)
            self.tracking_enabled = True
        else:
            self.detector = detector if isinstance(detector, PlayerDetector) else PlayerDetector(detector)
            self.tracking_enabled = False
        
        self.pose_estimator = pose_estimator
        self.pose_analyzer = PoseAnalyzer(pose_estimator)
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'total_poses': 0,
            'processing_time': 0.0
        }
        
        logger.info("✅ Video Processor initialized")
        if self.tracking_enabled:
            logger.info("   📍 Player tracking enabled")
    
    def process_frame(self, frame: np.ndarray, 
                     frame_idx: int) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Process single frame
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            
        Returns:
            Tuple of (annotated_frame, detections, poses)
        """
        # Detect players
        detections = self.detector.detect_players(frame)
        
        # Update tracking if enabled
        if self.tracking_enabled:
            detections = self.detector.update_tracking(
                detections, 
                TRACKING_CONFIG['max_distance']
            )
        
        # Process each detection
        poses = []
        annotated_frame = frame.copy()
        
        for i, det in enumerate(detections):
            # Crop player region
            cropped = self.detector.crop_player(frame, det['bbox'], padding=10)
            
            if cropped is None or cropped.size == 0:
                poses.append(None)
                continue
            
            # Estimate pose
            pose = self.pose_estimator.estimate_pose(cropped)
            
            if pose is not None:
                # Adjust keypoints to original frame coordinates
                pose = self._adjust_keypoints_to_frame(pose, det['bbox'])
                
                # Detect action
                action = self.pose_analyzer.detect_action(pose)
                pose['action'] = action
                
                # Add player ID if tracking
                if self.tracking_enabled and 'player_id' in det:
                    pose['player_id'] = det['player_id']
                
                # Draw pose on frame
                annotated_frame = self.pose_estimator.draw_pose(
                    annotated_frame, pose,
                    draw_keypoints=VISUALIZATION_CONFIG['draw_keypoints'],
                    draw_skeleton=VISUALIZATION_CONFIG['draw_skeleton']
                )
            
            poses.append(pose)
        
        # Draw bounding boxes
        if VISUALIZATION_CONFIG['draw_bboxes']:
            annotated_frame = self._draw_detections_with_info(
                annotated_frame, detections, poses
            )
        
        # Add frame info
        annotated_frame = self._add_frame_info(
            annotated_frame, frame_idx, len(detections), len([p for p in poses if p])
        )
        
        return annotated_frame, detections, poses
    
    def _adjust_keypoints_to_frame(self, pose: Dict, bbox: List[int]) -> Dict:
        """
        Adjust keypoints coordinates from cropped image to original frame
        
        Args:
            pose: Pose dictionary
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Adjusted pose dictionary
        """
        if pose is None:
            return None
        
        adjusted_pose = pose.copy()
        keypoints = np.array(pose['keypoints'])
        
        x1, y1, x2, y2 = bbox
        
        # Adjust coordinates
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1
        
        adjusted_pose['keypoints'] = keypoints.tolist()
        
        return adjusted_pose
    
    def _draw_detections_with_info(self, frame: np.ndarray,
                                   detections: List[Dict],
                                   poses: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes with additional information
        
        Args:
            frame: Input frame
            detections: List of detections
            poses: List of poses
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        bbox_color = VISUALIZATION_CONFIG['bbox_color']
        bbox_thickness = VISUALIZATION_CONFIG['bbox_thickness']
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), bbox_color, bbox_thickness)
            
            # Prepare label
            if self.tracking_enabled and 'player_id' in det:
                label = f"P{det['player_id']}"
            else:
                label = f"P{i+1}"
            
            # Add pose info if available
            if i < len(poses) and poses[i] is not None:
                pose = poses[i]
                visible = pose['visible_keypoints']
                action = pose.get('action', 'unknown')
                label += f" | {visible}kp | {action}"
            
            # Draw label with background
            output = add_text_with_background(
                output, label, (x1, y1 - 10),
                font_scale=0.5,
                thickness=1,
                text_color=(255, 255, 255),
                bg_color=bbox_color,
                padding=3
            )
        
        return output
    
    def _add_frame_info(self, frame: np.ndarray, 
                       frame_idx: int,
                       num_detections: int,
                       num_poses: int) -> np.ndarray:
        """
        Add frame information overlay
        
        Args:
            frame: Input frame
            frame_idx: Frame index
            num_detections: Number of detections
            num_poses: Number of valid poses
            
        Returns:
            Frame with info overlay
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Create info text
        info_lines = [
            f"Frame: {frame_idx}",
            f"Players: {num_detections}",
            f"Poses: {num_poses}"
        ]
        
        # Draw semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)
        
        # Draw text
        y_offset = 35
        for line in info_lines:
            cv2.putText(output, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 25
        
        return output
    
    def process_video(self, video_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     save_data: bool = True,
                     show_preview: bool = False) -> Dict:
        """
        Process entire video
        
        Args:
            video_path: Input video path
            output_path: Output video path (None for auto-generate)
            save_data: Whether to save pose data
            show_preview: Whether to show live preview
            
        Returns:
            Dictionary with processing results and statistics
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if not is_video_file(video_path):
            raise ValueError(f"File is not a valid video: {video_path}")
        
        logger.info("="*60)
        logger.info(f"🎬 Starting video processing: {video_path.name}")
        logger.info("="*60)
        
        # Generate output path if not provided
        if output_path is None:
            output_path = get_output_filename(video_path, "_pose_estimation", ".mp4")
        else:
            output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        
        # Open video
        with VideoReader(video_path) as reader:
            # Get video properties
            fps = VIDEO_CONFIG['fps'] if VIDEO_CONFIG['fps'] else reader.fps
            width = VIDEO_CONFIG['resize_width'] or reader.width
            height = VIDEO_CONFIG['resize_height'] or reader.height
            
            # Initialize video writer
            if EXPORT_CONFIG['save_video']:
                writer = VideoWriter(
                    output_path, fps, width, height,
                    codec=VIDEO_CONFIG['codec']
                )
            else:
                writer = None
            
            # Initialize progress tracker
            progress = ProgressTracker(reader.frame_count, desc="Processing frames")
            
            # Storage for data export
            all_poses_data = []
            
            # Process frames
            frame_idx = 0
            frame_skip = VIDEO_CONFIG['frame_skip']
            
            while True:
                ret, frame = reader.read_frame()
                
                if not ret:
                    break
                
                # Skip frames if configured
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue
                
                # Resize if needed
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                # Process frame
                annotated_frame, detections, poses = self.process_frame(frame, frame_idx)
                
                # Update statistics
                self.stats['total_detections'] += len(detections)
                self.stats['total_poses'] += len([p for p in poses if p])
                self.stats['processed_frames'] += 1
                
                # Save frame data
                if save_data:
                    all_poses_data.append({
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'detections': detections,
                        'poses': poses
                    })
                
                # Write to output video
                if writer:
                    writer.write_frame(annotated_frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Pose Estimation', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Processing stopped by user")
                        break
                
                # Update progress
                progress.update()
                frame_idx += 1
            
            # Clean up
            progress.close()
            if writer:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.stats['processing_time'] = processing_time
        self.stats['total_frames'] = frame_idx
        
        # Save data files
        if save_data and all_poses_data:
            self._save_processing_data(all_poses_data, video_path, output_path)
        
        # Print summary
        self._print_summary(video_path, output_path)
        
        # Return results
        results = {
            'input_path': str(video_path),
            'output_path': str(output_path),
            'statistics': self.stats.copy(),
            'processing_time': processing_time,
            'fps': self.stats['processed_frames'] / processing_time if processing_time > 0 else 0
        }
        
        return results
    
    def _save_processing_data(self, poses_data: List[Dict],
                             video_path: Path,
                             output_path: Path):
        """
        Save processing data to JSON and CSV
        
        Args:
            poses_data: List of frame data
            video_path: Input video path
            output_path: Output video path
        """
        logger.info("💾 Saving processing data...")
        
        # Create data directory
        data_dir = ensure_output_dir("data")
        base_name = video_path.stem
        
        # Save JSON
        if EXPORT_CONFIG['save_json']:
            json_path = data_dir / f"{base_name}_poses.json"
            
            # Prepare full data
            full_data = {
                'video_info': {
                    'input_path': str(video_path),
                    'output_path': str(output_path),
                    'total_frames': self.stats['total_frames'],
                    'processed_frames': self.stats['processed_frames']
                },
                'statistics': self.stats,
                'frames': poses_data
            }
            
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"   ✅ JSON: {json_path.name}")
        
        # Save CSV
        if EXPORT_CONFIG['save_csv']:
            csv_path = data_dir / f"{base_name}_poses.csv"
            save_poses_csv(poses_data, csv_path)
            logger.info(f"   ✅ CSV: {csv_path.name}")
    
    def _print_summary(self, video_path: Path, output_path: Path):
        """Print processing summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 Processing Summary")
        logger.info("="*60)
        logger.info(f"📹 Input video: {video_path.name}")
        logger.info(f"📤 Output video: {output_path.name}")
        logger.info(f"🎞️  Total frames: {self.stats['total_frames']}")
        logger.info(f"✅ Processed frames: {self.stats['processed_frames']}")
        logger.info(f"👥 Total detections: {self.stats['total_detections']}")
        logger.info(f"🧍 Total poses: {self.stats['total_poses']}")
        logger.info(f"⏱️  Processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['processing_time'] > 0:
            fps = self.stats['processed_frames'] / self.stats['processing_time']
            logger.info(f"⚡ Speed: {fps:.2f} FPS")
        
        logger.info("="*60 + "\n")
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'total_poses': 0,
            'processing_time': 0.0
        }
        
        if self.tracking_enabled:
            self.detector.reset_tracking()


class ImageProcessor:
    """
    Processor for single images or batch of images
    """
    
    def __init__(self, detector: PlayerDetector, 
                 pose_estimator: PoseEstimator):
        """
        Initialize Image Processor
        
        Args:
            detector: PlayerDetector instance
            pose_estimator: PoseEstimator instance
        """
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.pose_analyzer = PoseAnalyzer(pose_estimator)
        
        logger.info("✅ Image Processor initialized")
    
    def process_image(self, image_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None,
                     save_data: bool = True) -> Dict:
        """
        Process single image
        
        Args:
            image_path: Input image path
            output_path: Output image path (None for auto-generate)
            save_data: Whether to save pose data
            
        Returns:
            Dictionary with processing results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"🖼️  Processing image: {image_path.name}")
        
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Detect and crop players
        detections, cropped_images = self.detector.detect_and_crop_players(image)
        
        # Estimate poses
        poses = self.pose_estimator.estimate_poses_batch(cropped_images)
        
        # Create annotated image
        annotated = image.copy()
        
        # Draw detections and poses
        if VISUALIZATION_CONFIG['draw_bboxes']:
            annotated = self.detector.draw_detections(annotated, detections)
        
        for i, pose in enumerate(poses):
            if pose is not None:
                # Adjust keypoints to frame
                det = detections[i]
                pose = self._adjust_keypoints(pose, det['bbox'])
                
                # Draw pose
                annotated = self.pose_estimator.draw_pose(annotated, pose)
        
        # Generate output path
        if output_path is None:
            output_path = get_output_filename(image_path, "_pose_estimation")
        else:
            output_path = Path(output_path)
        
        # Save annotated image
        cv2.imwrite(str(output_path), annotated)
        logger.info(f"✅ Saved image: {output_path.name}")
        
        # Save data
        if save_data:
            json_path = output_path.with_suffix('.json')
            save_poses_json(poses, 0, json_path, {
                'image_path': str(image_path),
                'num_detections': len(detections)
            })
        
        results = {
            'input_path': str(image_path),
            'output_path': str(output_path),
            'num_detections': len(detections),
            'num_poses': len([p for p in poses if p]),
            'detections': detections,
            'poses': poses
        }
        
        return results
    
    def _adjust_keypoints(self, pose: Dict, bbox: List[int]) -> Dict:
        """
        Adjust keypoints coordinates from cropped image to original frame
        
        Args:
            pose: Pose dictionary
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Adjusted pose dictionary
        """
        if pose is None:
            return None
        
        adjusted_pose = pose.copy()
        keypoints = np.array(pose['keypoints'])
        
        x1, y1, x2, y2 = bbox
        
        # Adjust coordinates
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1
        
        adjusted_pose['keypoints'] = keypoints.tolist()
        
        return adjusted_pose
    
    def process_batch(self, image_dir: Union[str, Path],
                     output_dir: Optional[Union[str, Path]] = None,
                     save_data: bool = True) -> List[Dict]:
        """
        Process batch of images from directory
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory (None for default)
            save_data: Whether to save pose data
            
        Returns:
            List of processing results
        """
        image_dir = Path(image_dir)
        
        if not image_dir.exists() or not image_dir.is_dir():
            raise ValueError(f"Directory does not exist or is invalid: {image_dir}")
        
        # Get all image files
        from utils import get_supported_image_extensions
        image_extensions = get_supported_image_extensions()
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
        
        if not image_files:
            logger.warning(f"No images found in: {image_dir}")
            return []
        
        logger.info(f"📁 Processing {len(image_files)} images from: {image_dir.name}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = ensure_output_dir("images")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        results = []
        progress = ProgressTracker(len(image_files), desc="Processing images")
        
        for img_path in image_files:
            try:
                output_path = output_dir / f"{img_path.stem}_output{img_path.suffix}"
                result = self.process_image(img_path, output_path, save_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {str(e)}")
                results.append({
                    'input_path': str(img_path),
                    'error': str(e)
                })
            
            progress.update()
        
        progress.close()
        
        # Print summary
        successful = len([r for r in results if 'error' not in r])
        logger.info(f"\n✅ Successfully processed {successful}/{len(image_files)} images")
        
        return results


class BatchProcessor:
    """
    Advanced batch processor for multiple videos/images
    """
    
    def __init__(self, detector: PlayerDetector, 
                 pose_estimator: PoseEstimator,
                 enable_tracking: bool = True):
        """
        Initialize Batch Processor
        
        Args:
            detector: PlayerDetector instance
            pose_estimator: PoseEstimator instance
            enable_tracking: Whether to enable tracking for videos
        """
        self.video_processor = VideoProcessor(detector, pose_estimator, enable_tracking)
        self.image_processor = ImageProcessor(detector, pose_estimator)
        
        logger.info("✅ Batch Processor initialized")
    
    def process_directory(self, input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         process_videos: bool = True,
                         process_images: bool = True) -> Dict:
        """
        Process all videos and images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            process_videos: Whether to process videos
            process_images: Whether to process images
            
        Returns:
            Dictionary with processing results
        """
        input_dir = Path(input_dir)
        
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Directory does not exist or is invalid: {input_dir}")
        
        logger.info("="*60)
        logger.info(f"📂 Processing directory: {input_dir.name}")
        logger.info("="*60)
        
        # Setup output directory
        if output_dir is None:
            output_dir = OUTPUT_DIR / "batch_output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'videos': [],
            'images': []
        }
        
        # Process videos
        if process_videos:
            from utils import get_supported_video_extensions
            video_extensions = get_supported_video_extensions()
            video_files = []
            
            for ext in video_extensions:
                video_files.extend(input_dir.glob(f"*{ext}"))
            
            if video_files:
                logger.info(f"\n🎥 Found {len(video_files)} videos")
                
                video_output_dir = output_dir / "videos"
                video_output_dir.mkdir(exist_ok=True)
                
                for video_path in video_files:
                    try:
                        output_path = video_output_dir / f"{video_path.stem}_output.mp4"
                        result = self.video_processor.process_video(
                            video_path, output_path, save_data=True
                        )
                        results['videos'].append(result)
                        
                        # Reset stats for next video
                        self.video_processor.reset_stats()
                        
                    except Exception as e:
                        logger.error(f"Error processing video {video_path.name}: {str(e)}")
                        results['videos'].append({
                            'input_path': str(video_path),
                            'error': str(e)
                        })
        
        # Process images
        if process_images:
            from utils import get_supported_image_extensions
            image_extensions = get_supported_image_extensions()
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(input_dir.glob(f"*{ext}"))
            
            if image_files:
                logger.info(f"\n🖼️  Found {len(image_files)} images")
                
                image_output_dir = output_dir / "images"
                image_output_dir.mkdir(exist_ok=True)
                
                batch_results = self.image_processor.process_batch(
                    input_dir, image_output_dir, save_data=True
                )
                results['images'] = batch_results
        
        # Print final summary
        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: Dict):
        """Print batch processing summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 Overall processing summary")
        logger.info("="*60)
        
        # Videos summary
        videos = results['videos']
        if videos:
            successful_videos = len([v for v in videos if 'error' not in v])
            total_frames = sum(v.get('statistics', {}).get('processed_frames', 0) 
                             for v in videos if 'error' not in v)
            total_detections = sum(v.get('statistics', {}).get('total_detections', 0) 
                                 for v in videos if 'error' not in v)
            total_poses = sum(v.get('statistics', {}).get('total_poses', 0) 
                            for v in videos if 'error' not in v)
            
            logger.info(f"🎥 Videos:")
            logger.info(f"   - Processed: {successful_videos}/{len(videos)}")
            logger.info(f"   - Total frames: {total_frames}")
            logger.info(f"   - Total detections: {total_detections}")
            logger.info(f"   - Total poses: {total_poses}")
        
        # Images summary
        images = results['images']
        if images:
            successful_images = len([i for i in images if 'error' not in i])
            total_detections = sum(i.get('num_detections', 0) 
                                 for i in images if 'error' not in i)
            total_poses = sum(i.get('num_poses', 0) 
                            for i in images if 'error' not in i)
            
            logger.info(f"\n🖼️  Images:")
            logger.info(f"   - Processed: {successful_images}/{len(images)}")
            logger.info(f"   - Total detections: {total_detections}")
            logger.info(f"   - Total poses: {total_poses}")
        
        logger.info(f"\n📁 Output directory: {results['output_dir']}")
        logger.info("="*60 + "\n")


# Test function
if __name__ == "__main__":
    print("🧪 Testing Video Processor...\n")
    
    try:
        from models import initialize_models
        
        # Initialize models
        detector_model, pose_model, _ = initialize_models()
        
        # Create detector and pose estimator
        from detector import PlayerDetector
        from pose_estimator import PoseEstimator
        
        player_detector = PlayerDetector(detector_model)
        pose_estimator = PoseEstimator(pose_model)
        
        # Create processors
        video_processor = VideoProcessor(player_detector, pose_estimator)
        image_processor = ImageProcessor(player_detector, pose_estimator)
        batch_processor = BatchProcessor(player_detector, pose_estimator)
        
        print("\n✅ Video Processor tested successfully!")
        print("\nAvailable processors:")
        print("  - VideoProcessor: for processing videos")
        print("  - ImageProcessor: for processing images")
        print("  - BatchProcessor: for processing entire directories")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)
