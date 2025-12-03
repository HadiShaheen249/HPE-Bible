"""
Object Detector class for detecting players in football matches
Uses YOLO model to detect persons (players) in images and videos
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
import sys

# Import configuration
try:
    from config import (
        DETECTION_CONFIG,
        FILTER_CONFIG,
        VISUALIZATION_CONFIG,
        LOGGING_CONFIG
    )
except ImportError:
    print("❌ Error: config.py file not found")
    sys.exit(1)

# Setup logging
logger = logging.getLogger(__name__)


class PlayerDetector:
    """
    Class for detecting players (persons) in football match images/videos
    """
    
    def __init__(self, model):
        """
        Initialize Player Detector
        
        Args:
            model: DetectionModel instance from models.py
        """
        self.model = model
        self.config = DETECTION_CONFIG
        self.filter_config = FILTER_CONFIG
        self.viz_config = VISUALIZATION_CONFIG
        
        logger.info("✅ Player Detector initialized")
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect all players in a frame
        
        Args:
            frame: Input image/frame (numpy array)
            
        Returns:
            List of dictionaries containing detection info:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class_id': int,
                    'class_name': str,
                    'center': (cx, cy),
                    'area': float
                },
                ...
            ]
        """
        results = self.model.predict(frame)
        
        if results is None or results.boxes is None:
            return []
        
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            confidence = float(boxes.conf[i].cpu().numpy())
            
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = results.names[class_id]
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            area = (x2 - x1) * (y2 - y1)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name,
                'center': (cx, cy),
                'area': area
            }
            
            detections.append(detection)
        
        filtered_detections = self._filter_detections(detections)
        
        logger.debug(f"Detected {len(filtered_detections)} players in frame")
        
        return filtered_detections
    
    def _filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections based on area and other criteria
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        min_area = self.filter_config['min_bbox_area']
        max_area = self.filter_config['max_bbox_area']
        
        for det in detections:
            area = det['area']
            
            if min_area <= area <= max_area:
                filtered.append(det)
            else:
                logger.debug(f"Filtered detection with area {area} (out of range)")
        
        return filtered
    
    def crop_player(self, frame: np.ndarray, bbox: List[int], 
                    padding: int = 10) -> Optional[np.ndarray]:
        """
        Crop a player region from frame with padding
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around bbox in pixels
            
        Returns:
            Cropped player image or None if invalid
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid cropping coordinates")
            return None
        
        cropped = frame[y1:y2, x1:x2]
        
        return cropped
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict],
                       draw_labels: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            draw_labels: Whether to draw labels
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        bbox_color = self.viz_config['bbox_color']
        bbox_thickness = self.viz_config['bbox_thickness']
        text_color = self.viz_config['text_color']
        text_scale = self.viz_config['text_scale']
        text_thickness = self.viz_config['text_thickness']
        text_font = self.viz_config['text_font']
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            
            cv2.rectangle(output, (x1, y1), (x2, y2), 
                         bbox_color, bbox_thickness)
            
            if draw_labels and self.viz_config['draw_labels']:
                label = f"Player {i+1}: {confidence:.2f}"
                
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, text_font, text_scale, text_thickness
                )
                
                cv2.rectangle(output, 
                            (x1, y1 - text_h - baseline - 5),
                            (x1 + text_w, y1),
                            bbox_color, -1)
                
                cv2.putText(output, label,
                           (x1, y1 - baseline - 5),
                           text_font, text_scale, text_color, 
                           text_thickness, cv2.LINE_AA)
            
            cx, cy = det['center']
            cv2.circle(output, (cx, cy), 5, (0, 255, 255), -1)
        
        return output
    
    def get_detection_stats(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with statistics
        """
        if not detections:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'avg_area': 0.0,
            }
        
        confidences = [d['confidence'] for d in detections]
        areas = [d['area'] for d in detections]
        
        stats = {
            'count': len(detections),
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_area': np.mean(areas),
            'min_area': np.min(areas),
            'max_area': np.max(areas),
        }
        
        return stats
    
    def detect_and_crop_players(self, frame: np.ndarray, 
                                padding: int = 10) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        Detect players and crop their regions
        
        Args:
            frame: Input frame
            padding: Padding for cropping
            
        Returns:
            Tuple of (detections, cropped_images)
        """
        detections = self.detect_players(frame)
        
        cropped_images = []
        for det in detections:
            cropped = self.crop_player(frame, det['bbox'], padding)
            if cropped is not None:
                cropped_images.append(cropped)
            else:
                cropped_images.append(None)
        
        return detections, cropped_images
    
    def non_max_suppression(self, detections: List[Dict], 
                           iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]


class MultiPlayerDetector:
    """
    Extended detector with tracking capabilities for multiple players
    """
    
    def __init__(self, model):
        """
        Initialize Multi-Player Detector
        
        Args:
            model: DetectionModel instance
        """
        self.detector = PlayerDetector(model)
        self.tracked_players = {}
        self.next_player_id = 1
        
        logger.info("✅ Multi-Player Detector with tracking initialized")
    
    def detect_players(self, frame):
        """Detect players using underlying detector"""
        return self.detector.detect_players(frame)
    
    def crop_player(self, frame, bbox, padding=10):
        """Crop player region"""
        return self.detector.crop_player(frame, bbox, padding)    

    def update_tracking(self, detections: List[Dict], 
                       max_distance: float = 50) -> List[Dict]:
        """
        Update player tracking based on new detections
        
        Args:
            detections: List of detection dictionaries
            max_distance: Maximum distance for matching
            
        Returns:
            List of detections with player IDs
        """
        updated_detections = []
        
        for det in detections:
            center = det['center']
            
            min_dist = float('inf')
            matched_id = None
            
            for player_id, tracked_center in self.tracked_players.items():
                dist = np.sqrt((center[0] - tracked_center[0])**2 + 
                             (center[1] - tracked_center[1])**2)
                
                if dist < min_dist and dist < max_distance:
                    min_dist = dist
                    matched_id = player_id
            
            if matched_id is not None:
                det['player_id'] = matched_id
                self.tracked_players[matched_id] = center
            else:
                det['player_id'] = self.next_player_id
                self.tracked_players[self.next_player_id] = center
                self.next_player_id += 1
            
            updated_detections.append(det)
        
        return updated_detections
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracked_players = {}
        self.next_player_id = 1
        logger.info("Tracking reset")


# Test function
if __name__ == "__main__":
    print("🧪 Testing Player Detector...\n")
    
    try:
        from models import initialize_models
        
        detector_model, _, _ = initialize_models()
        
        player_detector = PlayerDetector(detector_model)
        
        print("\n✅ Player Detector test successful!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)
