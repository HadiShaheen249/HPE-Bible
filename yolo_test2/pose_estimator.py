"""
Pose Estimator class for estimating player poses using keypoints
Processes cropped player images and extracts pose keypoints
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import sys
from yolo_test2.config import Config

# Import configuration
try:
    from yolo_test2.config import (
        POSE_CONFIG,
        YOLO_KEYPOINTS,
        SKELETON_CONNECTIONS,
        VISUALIZATION_CONFIG,
        FILTER_CONFIG,
        LOGGING_CONFIG
    )
except ImportError:
    print("❌ Error: config.py not found")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)


class PoseEstimator:
    """
    Class for estimating player poses and extracting keypoints
    """
    
    def __init__(self, model):
        """
        Initialize Pose Estimator
        
        Args:
            model: PoseModel instance from models.py
        """
        self.model = model
        self.config = POSE_CONFIG
        self.keypoints_map = YOLO_KEYPOINTS
        self.skeleton_connections = SKELETON_CONNECTIONS
        self.viz_config = VISUALIZATION_CONFIG
        self.filter_config = FILTER_CONFIG
        
        logger.info("✅ Pose Estimator initialized")
    
    def estimate_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estimate pose for a single player image
        
        Args:
            image: Input player image (cropped)
            
        Returns:
            Dictionary containing pose information:
            {
                'keypoints': [[x, y, confidence], ...],
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'visible_keypoints': int,
                'keypoint_names': dict
            }
        """
        results = self.model.predict(image)
        
        if results is None or results.keypoints is None:
            logger.debug("No pose detected in image")
            return None
        
        keypoints_data = results.keypoints
        
        if len(keypoints_data) == 0:
            return None
        
        kpts = keypoints_data[0].data.cpu().numpy()[0]
        
        visible_count = np.sum(kpts[:, 2] > self.config['confidence_threshold'])
        
        if visible_count < self.filter_config['min_keypoints_visible']:
            logger.debug(f"Visible keypoints ({visible_count}) below minimum")
            return None
        
        bbox = None
        if results.boxes is not None and len(results.boxes) > 0:
            box = results.boxes[0].xyxy[0].cpu().numpy()
            bbox = [int(x) for x in box]
        
        confidence = float(results.boxes[0].conf[0].cpu().numpy()) if results.boxes is not None else 0.0
        
        pose_data = {
            'keypoints': kpts.tolist(),
            'bbox': bbox,
            'confidence': confidence,
            'visible_keypoints': int(visible_count),
            'keypoint_names': self.keypoints_map,
            'total_keypoints': len(kpts)
        }
        
        return pose_data
    
    def estimate_poses_batch(self, images: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Estimate poses for multiple player images
        """
        poses = []
        
        for i, img in enumerate(images):
            if img is None:
                poses.append(None)
                continue
            
            pose = self.estimate_pose(img)
            poses.append(pose)
            
            if pose:
                logger.debug(f"Pose {i+1}: {pose['visible_keypoints']} visible keypoints")
        
        return poses
    
    def draw_pose(self, image: np.ndarray, pose_data: Dict,
                  draw_keypoints: bool = True,
                  draw_skeleton: bool = True) -> np.ndarray:
        """
        Draw pose keypoints and skeleton on image
        """
        output = image.copy()
        
        if pose_data is None:
            return output
        
        keypoints = np.array(pose_data['keypoints'])
        conf_threshold = self.config['confidence_threshold']
        
        kpt_color = self.viz_config['keypoint_color']
        skel_color = self.viz_config['skeleton_color']
        kpt_radius = self.viz_config['keypoint_radius']
        skel_thickness = self.viz_config['skeleton_thickness']
        
        if draw_skeleton and self.viz_config['draw_skeleton']:
            for connection in self.skeleton_connections:
                idx1, idx2 = connection
                
                kpt1 = keypoints[idx1]
                kpt2 = keypoints[idx2]
                
                if kpt1[2] > conf_threshold and kpt2[2] > conf_threshold:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    
                    cv2.line(output, pt1, pt2, skel_color, skel_thickness, cv2.LINE_AA)
        
        if draw_keypoints and self.viz_config['draw_keypoints']:
            for i, kpt in enumerate(keypoints):
                x, y, conf = kpt
                
                if conf > conf_threshold:
                    center = (int(x), int(y))
                    
                    cv2.circle(output, center, kpt_radius, kpt_color, -1, cv2.LINE_AA)
                    cv2.circle(output, center, kpt_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        return output
    
    def get_keypoint_by_name(self, pose_data: Dict, keypoint_name: str) -> Optional[Tuple[float, float, float]]:
        """
        Get specific keypoint by name
        """
        if pose_data is None:
            return None
        
        keypoint_idx = None
        for idx, name in self.keypoints_map.items():
            if name == keypoint_name:
                keypoint_idx = idx
                break
        
        if keypoint_idx is None:
            logger.warning(f"Keypoint '{keypoint_name}' not found")
            return None
        
        keypoints = pose_data['keypoints']
        if keypoint_idx >= len(keypoints):
            return None
        
        kpt = keypoints[keypoint_idx]
        return tuple(kpt)
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        """
        p1 = np.array(p1[:2])
        p2 = np.array(p2[:2])
        p3 = np.array(p3[:2])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def get_joint_angles(self, pose_data: Dict) -> Dict[str, float]:
        """
        Calculate important joint angles
        """
        if pose_data is None:
            return {}
        
        angles = {}
        
        try:
            l_shoulder = self.get_keypoint_by_name(pose_data, 'left_shoulder')
            l_elbow = self.get_keypoint_by_name(pose_data, 'left_elbow')
            l_wrist = self.get_keypoint_by_name(pose_data, 'left_wrist')
            
            if all(kpt and kpt[2] > 0.5 for kpt in [l_shoulder, l_elbow, l_wrist]):
                angles['left_elbow'] = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            
            r_shoulder = self.get_keypoint_by_name(pose_data, 'right_shoulder')
            r_elbow = self.get_keypoint_by_name(pose_data, 'right_elbow')
            r_wrist = self.get_keypoint_by_name(pose_data, 'right_wrist')
            
            if all(kpt and kpt[2] > 0.5 for kpt in [r_shoulder, r_elbow, r_wrist]):
                angles['right_elbow'] = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            
            l_hip = self.get_keypoint_by_name(pose_data, 'left_hip')
            l_knee = self.get_keypoint_by_name(pose_data, 'left_knee')
            l_ankle = self.get_keypoint_by_name(pose_data, 'left_ankle')
            
            if all(kpt and kpt[2] > 0.5 for kpt in [l_hip, l_knee, l_ankle]):
                angles['left_knee'] = self.calculate_angle(l_hip, l_knee, l_ankle)
            
            r_hip = self.get_keypoint_by_name(pose_data, 'right_hip')
            r_knee = self.get_keypoint_by_name(pose_data, 'right_knee')
            r_ankle = self.get_keypoint_by_name(pose_data, 'right_ankle')
            
            if all(kpt and kpt[2] > 0.5 for kpt in [r_hip, r_knee, r_ankle]):
                angles['right_knee'] = self.calculate_angle(r_hip, r_knee, r_ankle)
        
        except Exception as e:
            logger.warning(f"Error calculating angles: {str(e)}")
        
        return angles
    
    def get_pose_stats(self, pose_data: Dict) -> Dict:
        """
        Get pose statistics
        """
        if pose_data is None:
            return {
                'total_keypoints': 0,
                'visible_keypoints': 0,
                'visibility_ratio': 0.0,
                'avg_confidence': 0.0,
                'joint_angles': {}
            }
        
        keypoints = np.array(pose_data['keypoints'])
        confidences = keypoints[:, 2]
        
        visible = np.sum(confidences > self.config['confidence_threshold'])
        
        stats = {
            'total_keypoints': pose_data['total_keypoints'],
            'visible_keypoints': int(visible),
            'visibility_ratio': visible / pose_data['total_keypoints'],
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'joint_angles': self.get_joint_angles(pose_data)
        }
        
        return stats
    
    def normalize_keypoints(self, pose_data: Dict, reference_size=(256, 256)) -> Dict:
        """
        Normalize keypoints to reference size
        """
        if pose_data is None or pose_data['bbox'] is None:
            return pose_data
        
        normalized_pose = pose_data.copy()
        keypoints = np.array(pose_data['keypoints'])
        bbox = pose_data['bbox']
        
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        normalized_kpts = keypoints.copy()
        normalized_kpts[:, 0] = (keypoints[:, 0] - x1) / bbox_w * reference_size[0]
        normalized_kpts[:, 1] = (keypoints[:, 1] - y1) / bbox_h * reference_size[1]
        
        normalized_pose['keypoints'] = normalized_kpts.tolist()
        normalized_pose['normalized'] = True
        normalized_pose['reference_size'] = reference_size
        
        return normalized_pose


class PoseAnalyzer:
    """
    Extended pose analyzer with action recognition capabilities
    """
    
    def __init__(self, pose_estimator: PoseEstimator):
        """
        Initialize Pose Analyzer
        """
        self.estimator = pose_estimator
        logger.info("✅ Pose Analyzer initialized")
    
    def detect_action(self, pose_data: Dict) -> str:
        """
        Detect action based on pose
        """
        if pose_data is None:
            return "unknown"
        
        angles = self.estimator.get_joint_angles(pose_data)
        
        action = "standing"
        
        try:
            if 'left_knee' in angles and angles['left_knee'] > 160:
                action = "kicking_left"
            elif 'right_knee' in angles and angles['right_knee'] > 160:
                action = "kicking_right"
            
            elif 'left_knee' in angles and 'right_knee' in angles:
                if 90 < angles['left_knee'] < 140 and 90 < angles['right_knee'] < 140:
                    action = "running"
            
            elif 'left_knee' in angles and 'right_knee' in angles:
                if angles['left_knee'] > 160 and angles['right_knee'] > 160:
                    action = "jumping"
        
        except Exception as e:
            logger.debug(f"Error detecting action: {str(e)}")
        
        return action
    
    def compare_poses(self, pose1: Dict, pose2: Dict) -> float:
        """
        Compare two poses and calculate similarity
        """
        if pose1 is None or pose2 is None:
            return 0.0
        
        kpts1 = np.array(pose1['keypoints'])
        kpts2 = np.array(pose2['keypoints'])
        
        distances = []
        conf_threshold = self.estimator.config['confidence_threshold']
        
        for i in range(len(kpts1)):
            if kpts1[i][2] > conf_threshold and kpts2[i][2] > conf_threshold:
                dist = np.linalg.norm(kpts1[i][:2] - kpts2[i][:2])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        avg_distance = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_distance)
        
        return similarity


if __name__ == "__main__":
    print("🧪 Testing Pose Estimator...\n")
    
    try:
        from models import initialize_models
        
        _, pose_model, _ = initialize_models()
        
        pose_estimator = PoseEstimator(pose_model)
        
        print("\n✅ Pose Estimator test successful!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)
