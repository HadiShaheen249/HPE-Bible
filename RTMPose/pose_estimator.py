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

# --- Essential Imports ---
from mmengine.registry import init_default_scope
import mmdet.datasets
from mmdet.utils import register_all_modules as register_mmdet_modules
from mmpose.utils import register_all_modules as register_mmpose_modules

# Ensure PackDetInputs is imported
try:
    from mmdet.datasets.transforms import PackDetInputs
except ImportError:
    pass
# -------------------------

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
            
            # --- Load Detection (RTMDet) ---
            det_config, det_checkpoint = Config.get_det_model_path()
            
            if not det_checkpoint.exists():
                print(f"📥 Detection checkpoint not found, downloading...")
                self._download_det_model()
            
            if not det_config.exists():
                print(f"📥 Using detection config from mim...")
                det_config = f'rtmdet_{self.det_model_name.split("-")[1]}_8xb32-300e_coco.py'
            
            print(f"📦 Loading detection model: {self.det_model_name}")
            
            # FORCE MMDET SCOPE
            register_mmdet_modules(init_default_scope=True)
            
            self.det_model = init_detector(
                str(det_config),
                str(det_checkpoint),
                device=self.device
            )
            print(f"✅ Detection model loaded!")
            
            # --- Load Pose (RTMPose) ---
            pose_config, pose_checkpoint = Config.get_pose_model_path()
            
            if not pose_checkpoint.exists():
                print(f"📥 Pose checkpoint not found, downloading...")
                self._download_pose_model()
            
            if not pose_config.exists():
                print(f"📥 Using pose config from mim...")
                pose_config = f'rtmpose-{self.pose_model_name.split("-")[1]}_8xb256-420e_coco-256x192.py'
            
            print(f"📦 Loading pose model: {self.pose_model_name}")
            
            # FORCE MMPOSE SCOPE
            register_mmpose_modules(init_default_scope=True)
            
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
            raise
    
    def _init_tracker(self):
        """Initialize ByteTrack tracker"""
        try:
            from byte_tracker import BYTETracker
            
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
            self.tracker = None
        except Exception as e:
            print(f"⚠️  Error initializing tracker: {e}")
            self.tracker = None
    
    def _download_det_model(self):
        from utils import ModelDownloader
        det_config, det_checkpoint = Config.get_det_model_path()
        if 'rtmdet-m' in self.det_model_name:
            checkpoint_url = Config.MODEL_URLS['rtmdet-m']['checkpoint']
            ModelDownloader.download_file(checkpoint_url, det_checkpoint)
        print(f"✅ Detection model downloaded!")
    
    def _download_pose_model(self):
        from utils import ModelDownloader
        pose_config, pose_checkpoint = Config.get_pose_model_path()
        if 'rtmpose-m' in self.pose_model_name:
            checkpoint_url = Config.MODEL_URLS['rtmpose-m']['checkpoint']
            ModelDownloader.download_file(checkpoint_url, pose_checkpoint)
        print(f"✅ Pose model downloaded!")
    
    def _detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect persons in frame
        """
        from mmdet.apis import inference_detector
        
        # 🔥 CRITICAL FIX: Force Reset MMDET Registry Scope 🔥
        register_mmdet_modules(init_default_scope=True)
        
        result = inference_detector(self.det_model, frame)
        
        pred_instances = result.pred_instances
        person_mask = (pred_instances.labels == 0) & \
                     (pred_instances.scores >= self.det_conf)
        
        bboxes = pred_instances.bboxes[person_mask].cpu().numpy()
        scores = pred_instances.scores[person_mask].cpu().numpy()
        
        if len(bboxes) > 0:
            detections = np.concatenate([bboxes, scores[:, None]], axis=1)
        else:
            detections = np.empty((0, 5))
        
        return detections
    
    def _estimate_pose(self, frame: np.ndarray, 
                      bboxes: np.ndarray) -> List[Dict]:
        """
        Estimate pose for detected persons
        """
        from mmpose.apis import inference_topdown
        
        if len(bboxes) == 0:
            return []
        
        # 🔥 CRITICAL FIX: Force Reset MMPOSE Registry Scope 🔥
        register_mmpose_modules(init_default_scope=True)
        
        bboxes_xyxy = bboxes[:, :4]
        
        results = inference_topdown(
            self.pose_model,
            frame,
            bboxes_xyxy
        )
        
        return results
    
    def _track_objects(self, detections: np.ndarray, 
                      frame_id: int,
                      img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if self.tracker is None or len(detections) == 0:
            return None
        
        online_targets = self.tracker.update(
            detections,
            img_info=img_shape,
            img_size=img_shape
        )
        
        tracked = []
        for track in online_targets:
            tlwh = track.tlwh
            track_id = track.track_id
            score = track.score
            
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            
            tracked.append([x1, y1, x2, y2, score, track_id])
        
        return np.array(tracked) if tracked else None
    
    def _visualize_results(self, frame: np.ndarray,
                          pose_results: List[Dict],
                          tracked_boxes: Optional[np.ndarray] = None) -> np.ndarray:
        vis_frame = frame.copy()
        
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
        
        for result in pose_results:
            keypoints = result.pred_instances.keypoints[0]
            scores = result.pred_instances.keypoint_scores[0]
            
            vis_frame = VisualizationHelper.draw_skeleton(
                vis_frame,
                keypoints,
                scores,
                Config.SKELETON_LINKS,
                color=Config.SKELETON_COLOR,
                thickness=Config.SKELETON_THICKNESS,
                conf_threshold=self.pose_conf
            )
            
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
        print(f"📸 Processing image: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        detections = self._detect_persons(frame)
        print(f"👥 Detected {len(detections)} person(s)")
        
        if len(detections) == 0:
            print("⚠️  No persons detected")
            return frame
        
        pose_results = self._estimate_pose(frame, detections)
        result_frame = self._visualize_results(frame, pose_results)
        
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
        img_shape = (height, width)
        
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
                
                detections = self._detect_persons(frame)
                
                tracked_boxes = None
                if self.tracker is not None and len(detections) > 0:
                    tracked_boxes = self._track_objects(detections, frame_id, img_shape)
                    if tracked_boxes is not None:
                        detections = tracked_boxes[:, :5]
                
                pose_results = []
                if len(detections) > 0:
                    pose_results = self._estimate_pose(frame, detections)
                
                result_frame = self._visualize_results(frame, pose_results, tracked_boxes)
                
                current_fps = monitor.get_current_fps()
                result_frame = VisualizationHelper.put_fps_text(result_frame, current_fps)
                
                info_text = f'Frame: {frame_id} | Persons: {len(pose_results)}'
                result_frame = VisualizationHelper.put_info_text(result_frame, info_text, position=(10, 70))
                
                if writer is not None:
                    writer.write(result_frame)
                
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
            monitor.print_stats()

def test_estimator():
    print("🧪 Testing RTMPose Estimator...")
    Config.print_paths()
    try:
        estimator = RTMPoseEstimator()
        print("✅ Estimator ready!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_estimator()