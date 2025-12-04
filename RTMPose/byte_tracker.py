"""
ByteTrack Implementation
Multi-Object Tracking with ByteTrack
"""

import numpy as np
from collections import deque
import cv2

# ============================================
# ✅ FIXED: Import required libraries with proper error handling
# ============================================
try:
    import scipy
    import scipy.linalg
    SCIPY_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scipy not installed, Kalman filter may not work properly")
    print("Install with: pip install scipy")
    SCIPY_AVAILABLE = False

try:
    import lap
    LAP_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: lap not installed, using scipy as fallback")
    print("Install with: pip install lap")
    LAP_AVAILABLE = False
    
    # Fallback to scipy if lap not available
    if SCIPY_AVAILABLE:
        from scipy.optimize import linear_sum_assignment


class STrack:
    """Single target track"""
    
    shared_kalman = None  # Shared Kalman filter for all tracks
    track_id_count = 0
    
    def __init__(self, tlwh, score):
        """
        Initialize track
        
        Args:
            tlwh: Bounding box in tlwh format [top, left, width, height]
            score: Detection score
        """
        # Wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        
        self.score = score
        self.tracklet_len = 0
        self.track_id = 0
        self.state = 'new'
        
        self.start_frame = 0
        self.frame_id = 0
    
    def predict(self):
        """Predict next state using Kalman filter"""
        mean_state = self.mean.copy()
        if self.state != 'tracked':
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )
    
    @staticmethod
    def multi_predict(stracks):
        """Predict multiple tracks"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            
            for i, st in enumerate(stracks):
                if st.state != 'tracked':
                    multi_mean[i][7] = 0
            
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
    
    def activate(self, kalman_filter, frame_id):
        """Activate a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        
        self.tracklet_len = 0
        self.state = 'tracked'
        
        if frame_id == 1:
            self.is_activated = True
        
        self.frame_id = frame_id
        self.start_frame = frame_id
    
    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a lost track"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        
        if new_id:
            self.track_id = self.next_id()
        
        self.score = new_track.score
    
    def update(self, new_track, frame_id):
        """Update track with new detection"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        
        self.state = 'tracked'
        self.is_activated = True
        self.score = new_track.score
    
    @property
    def tlwh(self):
        """Get current position in tlwh format"""
        if self.mean is None:
            return self._tlwh.copy()
        
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        """Convert to tlbr format [top, left, bottom, right]"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert tlwh to xyah format [center_x, center_y, aspect_ratio, height]"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
    
    def to_xyah(self):
        """Convert current position to xyah format"""
        return self.tlwh_to_xyah(self.tlwh)
    
    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert tlbr to tlwh format"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
    
    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert tlwh to tlbr format"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret
    
    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.frame_id})'
    
    @staticmethod
    def next_id():
        """Get next track ID"""
        STrack.track_id_count += 1
        return STrack.track_id_count
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = 'lost'
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = 'removed'


class BYTETracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    """
    
    def __init__(self, args, frame_rate=30):
        """
        Initialize ByteTrack
        
        Args:
            args: Configuration dictionary
            frame_rate: Video frame rate
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0
        
        # Parse args
        if isinstance(args, dict):
            self.track_thresh = args.get('track_thresh', 0.5)
            self.track_buffer = args.get('track_buffer', 30)
            self.match_thresh = args.get('match_thresh', 0.8)
            self.min_box_area = args.get('min_box_area', 10)
            self.mot20 = args.get('mot20', False)
        else:
            self.track_thresh = args.track_thresh
            self.track_buffer = args.track_buffer
            self.match_thresh = args.match_thresh
            self.min_box_area = args.min_box_area
            self.mot20 = args.mot20
        
        self.buffer_size = int(frame_rate / 30.0 * self.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        # Initialize shared Kalman filter
        STrack.shared_kalman = self.kalman_filter
    
    def update(self, output_results, img_info=None, img_size=None):
        """
        Update tracker with new detections
        
        Args:
            output_results: Detection results (N, 5) [x1, y1, x2, y2, score]
            img_info: Image info
            img_size: Image size
            
        Returns:
            List of active tracks
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        
        # Filter by score
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        
        # Convert to tlwh format
        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) 
                         for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
        
        # Add newly detected tracklets
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        # Step 2: First association, with high score detection boxes
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        
        # Predict current location with KF
        STrack.multi_predict(strack_pool)
        
        dists = matching.iou_distance(strack_pool, detections)
        
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == 'tracked':
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        # Step 3: Second association, with low score detection boxes
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) 
                               for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        
        r_tracked_stracks = [strack_pool[i] for i in u_track 
                            if strack_pool[i].state == 'tracked']
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == 'lost':
                track.mark_lost()
                lost_stracks.append(track)
        
        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        
        if not self.mot20:
            dists = matching.fuse_score(dists, detections)
        
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.track_thresh:
                continue
            
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        self.tracked_stracks = [t for t in self.tracked_stracks 
                               if t.state == 'tracked']
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        
        # Get current output
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks


class KalmanFilter:
    """
    Kalman Filter for object tracking
    """
    
    def __init__(self):
        """Initialize Kalman Filter"""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for KalmanFilter. Install with: pip install scipy")
        
        ndim, dt = 4, 1.
        
        # Create Kalman filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """
        Initialize track from unassociated measurement
        
        Args:
            measurement: Bounding box in format (x, y, a, h)
            
        Returns:
            mean: Mean vector
            covariance: Covariance matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step
        
        Args:
            mean: Mean vector
            covariance: Covariance matrix
            
        Returns:
            mean: Predicted mean
            covariance: Predicted covariance
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """
        Project state distribution to measurement space
        
        Args:
            mean: Mean vector
            covariance: Covariance matrix
            
        Returns:
            mean: Projected mean
            covariance: Projected covariance
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        ))
        
        return mean, covariance + innovation_cov
    
    # ============================================
    # ✅ FIXED: Complete multi_predict method
    # ============================================
    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step for multiple tracks
        
        Args:
            mean: Mean vectors (N, 8)
            covariance: Covariance matrices (N, 8, 8)
            
        Returns:
            mean: Predicted means
            covariance: Predicted covariances
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        
        sqr = np.square(np.r_[std_pos, std_vel]).T
        
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step
        
        Args:
            mean: Predicted mean
            covariance: Predicted covariance
            measurement: Measurement vector
            
        Returns:
            mean: Updated mean
            covariance: Updated covariance
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements,
                       only_position=False, metric='maha'):
        """
        Compute gating distance between state distribution and measurements
        
        Args:
            mean: Mean vector
            covariance: Covariance matrix
            measurements: Measurement vectors (N, 4)
            only_position: Use only position for distance
            metric: Distance metric ('maha' or 'gaussian')
            
        Returns:
            distances: Gating distances
        """
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        d = measurements - mean
        
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True,
                check_finite=False, overwrite_b=True
            )
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('Invalid distance metric')


class matching:
    """Matching utilities for tracking"""
    
    @staticmethod
    def iou_distance(atracks, btracks):
        """
        Compute IOU distance between tracks
        
        Args:
            atracks: Track list A
            btracks: Track list B
            
        Returns:
            cost_matrix: IOU distance matrix
        """
        if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
           (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
            atlbrs = atracks
            btlbrs = btracks
        else:
            atlbrs = [track.tlbr for track in atracks]
            btlbrs = [track.tlbr for track in btracks]
        
        ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
        if ious.size == 0:
            return ious
        
        ious = bbox_ious(
            np.ascontiguousarray(atlbrs, dtype=np.float32),
            np.ascontiguousarray(btlbrs, dtype=np.float32)
        )
        
        cost_matrix = 1 - ious
        
        return cost_matrix
    
    @staticmethod
    def fuse_score(cost_matrix, detections):
        """
        Fuse cost matrix with detection scores
        
        Args:
            cost_matrix: Cost matrix
            detections: Detection list
            
        Returns:
            fused_cost: Fused cost matrix
        """
        if cost_matrix.size == 0:
            return cost_matrix
        
        iou_sim = 1 - cost_matrix
        det_scores = np.array([det.score for det in detections])
        det_scores = np.expand_dims(det_scores, axis=0).repeat(
            cost_matrix.shape[0], axis=0
        )
        
        fuse_sim = iou_sim * det_scores
        fuse_cost = 1 - fuse_sim
        
        return fuse_cost
    
    @staticmethod
    def linear_assignment(cost_matrix, thresh):
        """
        Linear assignment with Hungarian algorithm
        
        Args:
            cost_matrix: Cost matrix
            thresh: Threshold for matching
            
        Returns:
            matches: Matched pairs
            unmatched_a: Unmatched from A
            unmatched_b: Unmatched from B
        """
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1]))
            )
        
        matches, unmatched_a, unmatched_b = [], [], []
        
        # ============================================
        # ✅ FIXED: Better linear assignment implementation
        # ============================================
        if LAP_AVAILABLE:
            # Use lap library (faster)
            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
            
            for ix, mx in enumerate(x):
                if mx >= 0:
                    matches.append([ix, mx])
            
            unmatched_a = np.where(x < 0)[0]
            unmatched_b = np.where(y < 0)[0]
            
        elif SCIPY_AVAILABLE:
            # Fallback to scipy
            cost_matrix_masked = cost_matrix.copy()
            cost_matrix_masked[cost_matrix_masked > thresh] = thresh + 1e5
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix_masked)
            
            # Filter matches by threshold
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] <= thresh:
                    matches.append([i, j])
                else:
                    unmatched_a.append(i)
                    unmatched_b.append(j)
            
            # Find all unmatched
            matched_rows = set([m[0] for m in matches])
            matched_cols = set([m[1] for m in matches])
            
            unmatched_a = list(set(range(cost_matrix.shape[0])) - matched_rows)
            unmatched_b = list(set(range(cost_matrix.shape[1])) - matched_cols)
            
        else:
            raise ImportError("Either 'lap' or 'scipy' is required for linear assignment")
        
        matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
        
        return matches, unmatched_a, unmatched_b


def bbox_ious(atlbrs, btlbrs):
    """
    Compute IOU between two sets of boxes
    
    Args:
        atlbrs: Boxes A in tlbr format (N, 4)
        btlbrs: Boxes B in tlbr format (M, 4)
        
    Returns:
        ious: IOU matrix (N, M)
    """
    ious = np.zeros((atlbrs.shape[0], btlbrs.shape[0]), dtype=np.float32)
    if ious.size == 0:
        return ious
    
    # Compute IOU
    for i, atlbr in enumerate(atlbrs):
        for j, btlbr in enumerate(btlbrs):
            # Intersection
            x1 = max(atlbr[0], btlbr[0])
            y1 = max(atlbr[1], btlbr[1])
            x2 = min(atlbr[2], btlbr[2])
            y2 = min(atlbr[3], btlbr[3])
            
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            
            # Union
            box1_area = (atlbr[2] - atlbr[0]) * (atlbr[3] - atlbr[1])
            box2_area = (btlbr[2] - btlbr[0]) * (btlbr[3] - btlbr[1])
            union_area = box1_area + box2_area - inter_area
            
            # IOU
            ious[i, j] = inter_area / union_area if union_area > 0 else 0
    
    return ious


def joint_stracks(tlista, tlistb):
    """
    Join two track lists
    
    Args:
        tlista: Track list A
        tlistb: Track list B
        
    Returns:
        Combined track list
    """
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """
    Subtract track list B from A
    
    Args:
        tlista: Track list A
        tlistb: Track list B
        
    Returns:
        Subtracted track list
    """
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """
    Remove duplicate tracks
    
    Args:
        stracksa: Track list A
        stracksb: Track list B
        
    Returns:
        Deduplicated track lists
    """
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    
    return resa, resb


if __name__ == "__main__":
    print("🧪 ByteTrack module loaded successfully!")
    
    # Test ByteTrack initialization
    config = {
        'track_thresh': 0.5,
        'track_buffer': 30,
        'match_thresh': 0.8,
        'min_box_area': 10,
        'mot20': False
    }
    
    try:
        tracker = BYTETracker(config, frame_rate=30)
        print(f"✅ ByteTrack initialized successfully!")
        print(f"📊 Track threshold: {tracker.track_thresh}")
        print(f"📊 Track buffer: {tracker.track_buffer}")
        print(f"📊 Match threshold: {tracker.match_thresh}")
    except Exception as e:
        print(f"❌ Error: {e}")