"""
RTMPose + ByteTrack Pose Estimator Class
Professional class for pose estimation with tracking
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import time
import torch

from config import Config
from utils import PerformanceMonitor, VisualizationHelper, FileManager


class RTMPoseEstimator:
    """
    Professional class for pose estimation using RTMPose + ByteTrack
    """
    
    def __init__(self, 
                 det_model_name: str = None,
                 pose_model_name: str = None,
                 det_conf: float = None,
                 pose_conf: float = None,
                 device: str = None):
        """
        Initialize the models
        
        Args:
            det_model_name: Detection model name
            pose_model_name: Pose model name
            det_conf: Detection confidence threshold
            pose_conf: Pose confidence threshold
            device: Device to use (cuda:0 or cpu)
        """
        # Set configurations
        self.det_model_name = det_model_name or Config.DET_MODEL_NAME
        self.pose_model_name = pose_model_name or Config.POSE_MODEL_NAME
        self.det_conf = det_conf or Config.DET_CONFIDENCE
        self.pose_conf = pose_conf or Config.POSE_CONFIDENCE
        self.device = device or Config.validate_device()
        
        # Initialize models
        self.det_model = None
        self.pose_model = None
        self.tracker = None
        
        # Load models
        self._load_models()
        
        print(f"✅ RTMPose Estimator initialized successfully!")
        print(f"📱 Device: {self.device}")
    
    def _load_models(self):
        """Load detection and pose estimation models"""
        try:
            from mmdet.apis import init_detector
            from mmpose.apis import init_model as init_pose_model
            
            print(f"🔄 Loading models...")
            
            # Load detection model (RTMDet)
            det_config, det_checkpoint = Config.get_det_model_path()
            
            # ============================================
            # ✅ FIXED: Better model loading with mim support
            # ============================================
            if not det_checkpoint.exists():
                print(f"📥 Detection checkpoint not found, downloading...")
                self._download_det_model()
            
            # Use mim to get config if not exists locally
            if not det_config.exists():
                print(f"📥 Using detection config from mim...")
                det_config = f'rtmdet_{self.det_model_name.split("-")[1]}_8xb32-300e_coco.py'
            
            print(f"📦 Loading detection model: {self.det_model_name}")
            self.det_model = init_detector(
                str(det_config),
                str(det_checkpoint),
                device=self.device
            )
            print(f"✅ Detection model loaded!")
            
            # Load pose model (RTMPose)
            pose_config, pose_checkpoint = Config.get_pose_model_path()
            
            if not pose_checkpoint.exists():
                print(f"📥 Pose checkpoint not found, downloading...")
                self._download_pose_model()
            
            # Use mim to get config if not exists locally
            if not pose_config.exists():
                print(f"📥 Using pose config from mim...")
                pose_config = f'rtmpose-{self.pose_model_name.split("-")[1]}_8xb256-420e_coco-256x192.py'
            
            print(f"📦 Loading pose model: {self.pose_model_name}")
            self.pose_model = init_pose_model(
                str(pose_config),
                str(pose_checkpoint),
                device=self.device
            )
            print(f"✅ Pose model loaded!")
            
            # Initialize ByteTrack
            self._init_tracker()
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(f"💡 Tip: Run the following commands:")
            print(f"   pip install openmim")
            print(f"   mim install mmengine")
            print(f"   mim install 'mmcv>=2.0.0'")
            print(f"   mim install 'mmdet>=3.0.0'")
            print(f"   mim install 'mmpose>=1.0.0'")
            raise
    
    def _init_tracker(self):
        """Initialize ByteTrack tracker"""
        try:
            from byte_tracker import BYTETracker
            
            # ByteTrack configuration
            tracker_config = {
                'track_thresh': Config.TRACK_THRESH,
                'track_buffer': Config.TRACK_BUFFER,
                'match_thresh': Config.MATCH_THRESH,
                'min_box_area': Config.MIN_BOX_AREA,
                'mot20': Config.MOT20,
                'frame_rate': Config.FRAME_RATE
            }
            
            self.tracker = BYTETracker(tracker_config)
            print(f"✅ ByteTrack initialized!")
            
        except ImportError as e:
            print(f"⚠️  ByteTrack not available: {e}")
            print(f"💡 Make sure byte_tracker.py is in the same directory")
            self.tracker = None
        except Exception as e:
            print(f"⚠️  Error initializing tracker: {e}")
            self.tracker = None
    
    def _download_det_model(self):
        """Download detection model files"""
        from utils import ModelDownloader
        
        det_config, det_checkpoint = Config.get_det_model_path()
        
        # Download checkpoint
        if 'rtmdet-m' in self.det_model_name:
            checkpoint_url = Config.MODEL_URLS['rtmdet-m']['checkpoint']
            ModelDownloader.download_file(checkpoint_url, det_checkpoint)
        
        print(f"✅ Detection model downloaded!")
    
    def _download_pose_model(self):
        """Download pose model files"""
        from utils import ModelDownloader
        
        pose_config, pose_checkpoint = Config.get_pose_model_path()
        
        # Download checkpoint
        if 'rtmpose-m' in self.pose_model_name:
            checkpoint_url = Config.MODEL_URLS['rtmpose-m']['checkpoint']
            ModelDownloader.download_file(checkpoint_url, pose_checkpoint)
        
        print(f"✅ Pose model downloaded!")
    
    def _detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect persons in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Array of bounding boxes [x1, y1, x2, y2, score]
        """
        from mmdet.apis import inference_detector
        
        result = inference_detector(self.det_model, frame)
        
        # Get person detections (class 0 in COCO)
        pred_instances = result.pred_instances
        
        # Filter by class (person) and confidence
        person_mask = (pred_instances.labels == 0) & \
                     (pred_instances.scores >= self.det_conf)
        
        bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
        scores = pred_instances.scores[person_mask].cpu().numpy()
        
        # Combine bboxes and scores
        if len(bboxes) > 0:
            detections = np.concatenate([bboxes, scores[:, None]], axis=1)
        else:
            detections = np.empty((0, 5))
        
        return detections
    
    def _estimate_pose(self, frame: np.ndarray, 
                      bboxes: np.ndarray) -> List[Dict]:
        """
        Estimate pose for detected persons
        
        Args:
            frame: Input frame
            bboxes: Bounding boxes (N, 5) [x1, y1, x2, y2, score]
            
        Returns:
            List of pose results
        """
        from mmpose.apis import inference_topdown
        
        if len(bboxes) == 0:
            return []
        
        # Prepare bboxes for MMPose
        bboxes_xyxy = bboxes[:, :4]
        
        # Inference
        results = inference_topdown(
            self.pose_model,
            frame,
            bboxes_xyxy
        )
        
        return results
    
    # ============================================
    # ✅ FIXED: _track_objects method with correct parameters
    # ============================================
    def _track_objects(self, detections: np.ndarray, 
                      frame_id: int,
                      img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Track detected objects using ByteTrack
        
        Args:
            detections: Detections (N, 5) [x1, y1, x2, y2, score]
            frame_id: Current frame ID
            img_shape: Image shape (height, width)
            
        Returns:
            Tracked objects with IDs (N, 6) [x1, y1, x2, y2, score, track_id]
        """
        if self.tracker is None or len(detections) == 0:
            return None
        
        # Update tracker
        online_targets = self.tracker.update(
            detections,
            img_info=img_shape,  # ✅ FIXED: Use img_shape parameter
            img_size=img_shape
        )
        
        # Extract tracked results
        tracked = []
        for track in online_targets:
            tlwh = track.tlwh
            track_id = track.track_id
            score = track.score
            
            # Convert to [x1, y1, x2, y2, score, track_id]
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            
            tracked.append([x1, y1, x2, y2, score, track_id])
        
        return np.array(tracked) if tracked else None
    
    def _visualize_results(self, frame: np.ndarray,
                          pose_results: List[Dict],
                          tracked_boxes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize pose estimation and tracking results
        
        Args:
            frame: Input frame
            pose_results: Pose estimation results
            tracked_boxes: Tracked bounding boxes with IDs
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Draw tracked bounding boxes
        if tracked_boxes is not None and len(tracked_boxes) > 0:
            for box in tracked_boxes:
                x1, y1, x2, y2, score, track_id = box
                vis_frame = VisualizationHelper.draw_bbox(
                    vis_frame,
                    (x1, y1, x2, y2),
                    track_id=int(track_id),
                    color=Config.BBOX_COLOR,
                    thickness=Config.BBOX_THICKNESS
                )
        
        # Draw pose keypoints and skeleton
        for result in pose_results:
            keypoints = result.pred_instances.keypoints[0]  # (17, 2)
            scores = result.pred_instances.keypoint_scores[0]  # (17,)
            
            # Draw skeleton
            vis_frame = VisualizationHelper.draw_skeleton(
                vis_frame,
                keypoints,
                scores,
                Config.SKELETON_LINKS,
                color=Config.SKELETON_COLOR,
                thickness=Config.SKELETON_THICKNESS,
                conf_threshold=self.pose_conf
            )
            
            # Draw keypoints
            vis_frame = VisualizationHelper.draw_keypoints(
                vis_frame,
                keypoints,
                scores,
                Config.KEYPOINT_COLORS,
                radius=Config.KEYPOINT_RADIUS,
                conf_threshold=self.pose_conf
            )
        
        return vis_frame
    
    def predict_image(self, 
                     image_path: Union[str, Path],
                     save_result: bool = True,
                     output_path: Optional[str] = None) -> np.ndarray:
        """
        Analyze a single image
        
        Args:
            image_path: Path to the image
            save_result: Whether to save the result
            output_path: Path to save the result (optional)
            
        Returns:
            Processed image
        """
        print(f"📸 Processing image: {image_path}")
        
        # Read image
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Detect persons
        detections = self._detect_persons(frame)
        print(f"👥 Detected {len(detections)} person(s)")
        
        if len(detections) == 0:
            print("⚠️  No persons detected")
            return frame
        
        # Estimate pose
        pose_results = self._estimate_pose(frame, detections)
        
        # Visualize
        result_frame = self._visualize_results(frame, pose_results)
        
        # Save result
        if save_result:
            if output_path is None:
                input_filename = Path(image_path).name
                output_path = Config.OUTPUT_IMAGES_DIR / f"output_{input_filename}"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), result_frame)
            print(f"💾 Result saved to: {output_path}")
        
        return result_frame
    
    def predict_video(self,
                     video_path: Union[str, Path, int],
                     save_result: bool = True,
                     output_path: Optional[str] = None,
                     show_live: bool = True) -> None:
        """
        Analyze a video with tracking
        
        Args:
            video_path: Path to the video or 0 for camera
            save_result: Whether to save the result
            output_path: Path to save the video (optional)
            show_live: Whether to display the result live
        """
        # Open video
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
        
        # Video settings
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # ✅ FIXED: Store image shape for tracker
        img_shape = (height, width)
        
        # Setup video writer
        writer = None
        if save_result:
            if output_path is None:
                if is_camera:
                    timestamp = FileManager.get_timestamp()
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
        
        # Performance monitor
        monitor = PerformanceMonitor()
        monitor.start()
        
        frame_id = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id += 1
                monitor.add_frame()
                
                # Detect persons
                detections = self._detect_persons(frame)
                
                # Track objects
                tracked_boxes = None
                if self.tracker is not None and len(detections) > 0:
                    # ✅ FIXED: Pass img_shape to tracker
                    tracked_boxes = self._track_objects(detections, frame_id, img_shape)
                    
                    # Use tracked boxes for pose estimation if available
                    if tracked_boxes is not None:
                        detections = tracked_boxes[:, :5]
                
                # Estimate pose
                pose_results = []
                if len(detections) > 0:
                    pose_results = self._estimate_pose(frame, detections)
                
                # Visualize
                result_frame = self._visualize_results(
                    frame, 
                    pose_results,
                    tracked_boxes
                )
                
                # Add FPS info
                current_fps = monitor.get_current_fps()
                result_frame = VisualizationHelper.put_fps_text(
                    result_frame, current_fps
                )
                
                # Add frame info
                info_text = f'Frame: {frame_id} | Persons: {len(pose_results)}'
                result_frame = VisualizationHelper.put_info_text(
                    result_frame, info_text, position=(10, 70)
                )
                
                # Save frame
                if writer is not None:
                    writer.write(result_frame)
                
                # Display result
                if show_live:
                    cv2.imshow('RTMPose + ByteTrack - Press Q to Exit', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("⏹️  Processing stopped by user")
                        break
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print(f"💾 Video saved successfully!")
            cv2.destroyAllWindows()
            
            # Print statistics
            monitor.print_stats()


def test_estimator():
    """Test the estimator"""
    print("🧪 Testing RTMPose Estimator...")
    
    Config.print_paths()
    Config.print_model_info()
    
    try:
        estimator = RTMPoseEstimator()
        print("✅ Estimator ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_estimator()