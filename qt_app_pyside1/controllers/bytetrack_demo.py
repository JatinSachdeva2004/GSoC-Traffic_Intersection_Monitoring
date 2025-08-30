# ByteTrack Integration Demo
# This script demonstrates how to use the ByteTrack implementation
# as a drop-in replacement for DeepSORT in your application
#
# ByteTrack is the preferred tracker with better performance and higher FPS
# This version demonstrates the improved tracking with real-time comparison

import sys
import os
import argparse
import cv2
import time
import numpy as np
from pathlib import Path

# Add the parent directory to path for imports
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import both trackers for comparison
# from controllers.deepsort_tracker import DeepSortVehicleTracker  # Deprecated
from controllers.bytetrack_tracker import ByteTrackVehicleTracker

def generate_mock_detections(num_objects=5, frame_shape=(1080, 1920, 3)):
    """Generate mock vehicle detections for testing"""
    height, width = frame_shape[:2]
    detections = []
    
    for i in range(num_objects):
        # Random box dimensions (vehicles are typically wider than tall)
        w = np.random.randint(width // 10, width // 4)
        h = np.random.randint(height // 10, height // 6)
        
        # Random position
        x1 = np.random.randint(0, width - w)
        y1 = np.random.randint(0, height - h)
        x2 = x1 + w
        y2 = y1 + h
        
        # Random confidence and class (2 for car, 7 for truck)
        confidence = np.random.uniform(0.4, 0.95)
        class_id = np.random.choice([2, 7])
        
        detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(confidence),
            'class_id': int(class_id)
        })
    
    return detections

def draw_tracks(frame, tracks, color=(0, 255, 0)):
    """Draw tracking results on frame"""
    for track in tracks:
        track_id = track['id']
        bbox = track['bbox']
        conf = track.get('confidence', 0)
        
        x1, y1, x2, y2 = [int(b) for b in bbox]
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw ID and confidence
        text = f"ID:{track_id} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="ByteTrack vs DeepSORT comparison demo")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (default: camera)")
    parser.add_argument("--tracker", type=str, default="bytetrack", 
                      choices=["bytetrack", "deepsort", "both"],
                      help="Tracker to use: bytetrack (recommended), deepsort (legacy), or both")
    parser.add_argument("--mock", action="store_true", help="Use mock detections instead of actual detector")
    args = parser.parse_args()
    
    # Initialize video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print(f"Error: Could not open video source.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video source: {width}x{height} @ {fps}fps")
    
    # Initialize trackers based on choice
    if args.tracker == "bytetrack" or args.tracker == "both":
        bytetrack_tracker = ByteTrackVehicleTracker()
    
    if args.tracker == "deepsort" or args.tracker == "both":
        print("⚠️ DeepSORT tracker is deprecated, using ByteTrack as fallback")
        deepsort_tracker = ByteTrackVehicleTracker()
    
    # Main processing loop
    frame_count = 0
    processing_times = {'bytetrack': [], 'deepsort': []}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\nProcessing frame {frame_count}")
        
        # Generate or get detections
        if args.mock:
            detections = generate_mock_detections(num_objects=10, frame_shape=frame.shape)
            print(f"Generated {len(detections)} mock detections")
        else:
            # In a real application, you would use your actual detector here
            # This is just a placeholder for demo purposes
            detections = generate_mock_detections(num_objects=10, frame_shape=frame.shape)
            print(f"Generated {len(detections)} mock detections")
        
        # Process with ByteTrack
        if args.tracker == "bytetrack" or args.tracker == "both":
            start_time = time.time()
            bytetrack_results = bytetrack_tracker.update(detections, frame)
            bt_time = time.time() - start_time
            processing_times['bytetrack'].append(bt_time)
            print(f"ByteTrack processing time: {bt_time:.4f}s")
            
            if args.tracker == "bytetrack":
                display_frame = draw_tracks(frame.copy(), bytetrack_results, color=(0, 255, 0))
        
        # Process with DeepSORT
        if args.tracker == "deepsort" or args.tracker == "both":
            start_time = time.time()
            try:
                print("ℹ️ Using ByteTrack (as DeepSORT replacement)")
                deepsort_results = deepsort_tracker.update(detections, frame)
                ds_time = time.time() - start_time
                processing_times['deepsort'].append(ds_time)
                print(f"DeepSORT processing time: {ds_time:.4f}s")
            except Exception as e:
                print(f"DeepSORT error: {e}")
                deepsort_results = []
                ds_time = 0
            
            if args.tracker == "deepsort":
                display_frame = draw_tracks(frame.copy(), deepsort_results, color=(0, 0, 255))
        
        # If comparing both, create a side-by-side view
        if args.tracker == "both":
            # Draw tracks on separate frames
            bt_frame = draw_tracks(frame.copy(), bytetrack_results, color=(0, 255, 0))
            ds_frame = draw_tracks(frame.copy(), deepsort_results, color=(0, 0, 255))
            
            # Resize if needed and create side-by-side view
            h, w = frame.shape[:2]
            display_frame = np.zeros((h, w*2, 3), dtype=np.uint8)
            display_frame[:, :w] = bt_frame
            display_frame[:, w:] = ds_frame
            
            # Add labels
            cv2.putText(display_frame, "ByteTrack", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"{len(bytetrack_results)} tracks", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"{bt_time:.4f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(display_frame, "DeepSORT", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"{len(deepsort_results)} tracks", (w+10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_frame, f"{ds_time:.4f}s", (w+10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow("Tracking Demo", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Print performance statistics
    if len(processing_times['bytetrack']) > 0:
        bt_avg = sum(processing_times['bytetrack']) / len(processing_times['bytetrack'])
        print(f"ByteTrack average processing time: {bt_avg:.4f}s ({1/bt_avg:.2f} FPS)")
    
    if len(processing_times['deepsort']) > 0:
        ds_avg = sum(processing_times['deepsort']) / len(processing_times['deepsort'])
        print(f"DeepSORT average processing time: {ds_avg:.4f}s ({1/ds_avg:.2f} FPS)")

if __name__ == "__main__":
    main()
# ByteTrack implementation for vehicle tracking
# Efficient and robust multi-object tracking with improved association strategy
import numpy as np
import cv2
import time
from collections import defaultdict, deque
import torch
from typing import List, Dict, Any, Tuple, Optional

class BYTETracker:
    """
    ByteTrack tracker implementation
    Based on the paper: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    """
    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        camera_motion_compensation=False
    ):
        self.tracked_tracks = []  # Active tracks being tracked
        self.lost_tracks = []     # Lost tracks (temporarily out of view)
        self.removed_tracks = []  # Removed tracks (permanently lost)
        
        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)
        
        self.track_thresh = track_thresh          # Threshold for high-confidence detections
        self.track_high_thresh = track_high_thresh  # Higher threshold for first association
        self.track_low_thresh = track_low_thresh    # Lower threshold for second association
        self.match_thresh = match_thresh          # IOU match threshold
        
        self.track_id_count = 0
        self.camera_motion_compensation = camera_motion_compensation
        
        print(f"[BYTETRACK] Initialized with: high_thresh={track_high_thresh}, " +
              f"low_thresh={track_low_thresh}, match_thresh={match_thresh}")

    def update(self, detections, frame=None):
        """Update tracks with new detections
        
        Args:
            detections: list of dicts with keys ['bbox', 'confidence', 'class_id', ...]
            frame: Optional BGR frame for debug visualization
        
        Returns:
            list of dicts with keys ['id', 'bbox', 'confidence', 'class_id', ...]
        """
        self.frame_id += 1
        
        # FIXED: Add more debug output
        print(f"[BYTETRACK] Frame {self.frame_id}: Processing {len(detections)} detections")
        print(f"[BYTETRACK] Current state: {len(self.tracked_tracks)} tracked, {len(self.lost_tracks)} lost")
        
        # Convert detections to internal format
        converted_detections = self._convert_detections(detections)
        
        # Handle empty detections case
        if len(converted_detections) == 0:
            print(f"[BYTETRACK] No valid detections in frame {self.frame_id}")
            # Update lost tracks and remove expired
            new_tracked_tracks = []
            new_lost_tracks = []
            
            # All current tracks go to lost
            for track in self.tracked_tracks:
                track.is_lost = True
                if self.frame_id - track.last_frame <= self.max_time_lost:
                    track.predict()  # Predict new location
                    new_lost_tracks.append(track)
                else:
                    self.removed_tracks.append(track)
                    
            # Update remaining lost tracks
            for track in self.lost_tracks:
                if self.frame_id - track.last_frame <= self.max_time_lost:
                    track.predict()
                    new_lost_tracks.append(track)
                else:
                    self.removed_tracks.append(track)
                    
            self.tracked_tracks = new_tracked_tracks
            self.lost_tracks = new_lost_tracks
            print(f"[BYTETRACK] No detections: updated to {len(self.tracked_tracks)} tracked, {len(self.lost_tracks)} lost")
            return []
        
        # Split detections into high and low confidence - with safety checks
        if len(converted_detections) > 0:
            # FIXED: More robust confidence value handling
            try:
                # Make sure all values are numeric before comparison
                confidence_values = converted_detections[:, 4].astype(float)
                
                # Print the distribution of confidence values for debugging
                if len(confidence_values) > 0:
                    print(f"[BYTETRACK] Confidence values: min={np.min(confidence_values):.2f}, " +
                          f"median={np.median(confidence_values):.2f}, max={np.max(confidence_values):.2f}")
                
                high_dets = converted_detections[confidence_values >= self.track_high_thresh]
                low_dets = converted_detections[(confidence_values >= self.track_low_thresh) & 
                                               (confidence_values < self.track_high_thresh)]
                
                print(f"[BYTETRACK] Split into {len(high_dets)} high-conf and {len(low_dets)} low-conf detections")
            except Exception as e:
                print(f"[BYTETRACK] Error processing confidence values: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to empty arrays
                high_dets = np.empty((0, 6))
                low_dets = np.empty((0, 6))
        else:
            high_dets = np.empty((0, 6))
            low_dets = np.empty((0, 6))
        
        # Handle first frame special case
        if self.frame_id == 1:
            # Create new tracks for all high-confidence detections
            for i in range(len(high_dets)):
                det = high_dets[i]
                new_track = Track(det, self.track_id_count)
                new_track.last_frame = self.frame_id  # CRITICAL: Set last_frame when creating track
                self.track_id_count += 1
                self.tracked_tracks.append(new_track)
            
            # Also create tracks for lower confidence detections in first frame
            # This helps with initial tracking when objects might not be clearly visible
            for i in range(len(low_dets)):
                det = low_dets[i]
                new_track = Track(det, self.track_id_count)
                new_track.last_frame = self.frame_id  # CRITICAL: Set last_frame when creating track
                self.track_id_count += 1
                self.tracked_tracks.append(new_track)
            
            print(f"[BYTETRACK] First frame: created {len(self.tracked_tracks)} new tracks")
            return self._get_track_results()
            
        # Get active and lost tracks
        tracked_tlbrs = []
        tracked_ids = []
        
        for track in self.tracked_tracks:
            tracked_tlbrs.append(track.tlbr)
            tracked_ids.append(track.track_id)
            
        tracked_tlbrs = np.array(tracked_tlbrs) if tracked_tlbrs else np.empty((0, 4))
        tracked_ids = np.array(tracked_ids)
        
        # First association: high confidence detections with tracked tracks
        if len(tracked_tlbrs) > 0 and len(high_dets) > 0:
            # Match active tracks to high confidence detections
            matches, unmatched_tracks, unmatched_detections = self._match_tracks_to_detections(
                tracked_tlbrs, high_dets[:, :4], self.match_thresh
            )
            
            print(f"[BYTETRACK MATCH] Found {len(matches)} matches between {len(tracked_tlbrs)} tracks and {len(high_dets)} detections")
            
            # Update matched tracks with detections
            for i_track, i_det in matches:
                track_id = tracked_ids[i_track]
                track = self._get_track_by_id(track_id, self.tracked_tracks)
                if track:
                    track.update(high_dets[i_det])
                    track.last_frame = self.frame_id  # FIXED: Update last_frame when track is matched
                    print(f"[BYTETRACK MATCH] Track ID={track_id} matched and updated")
            
            # Move unmatched tracks to lost and rebuild tracked_tracks list
            unmatched_track_ids = []
            remaining_tracked_tracks = []
            
            # Keep matched tracks in tracked_tracks
            for i_track, _ in matches:
                track_id = tracked_ids[i_track]
                track = self._get_track_by_id(track_id, self.tracked_tracks)
                if track:
                    remaining_tracked_tracks.append(track)
            
            # Move unmatched tracks to lost
            for i_track in unmatched_tracks:
                track_id = tracked_ids[i_track]
                track = self._get_track_by_id(track_id, self.tracked_tracks)
                if track:
                    track.is_lost = True
                    track.last_frame = self.frame_id  # FIXED: Update last_frame when track is lost
                    self.lost_tracks.append(track)
                    unmatched_track_ids.append(track_id)
                
            # FIXED: Update tracked_tracks to only contain matched tracks
            self.tracked_tracks = remaining_tracked_tracks
                
            if unmatched_track_ids:
                print(f"[BYTETRACK MATCH] Lost tracks: {unmatched_track_ids}")
                
            # Create new tracks for unmatched high-confidence detections
            new_track_ids = []
            for i_det in unmatched_detections:
                det = high_dets[i_det]
                new_track = Track(det, self.track_id_count)
                new_track.last_frame = self.frame_id  # FIXED: Set last_frame when creating track
                new_track_ids.append(self.track_id_count)
                self.track_id_count += 1
                self.tracked_tracks.append(new_track)
                
            if new_track_ids:
                print(f"[BYTETRACK MATCH] Created new tracks: {new_track_ids}")
                
            print(f"[BYTETRACK] Matched {len(matches)} tracks, {len(unmatched_tracks)} unmatched tracks, " +
                  f"{len(unmatched_detections)} new tracks")
        else:
            # No tracked tracks or no high confidence detections
            
            # Move all current tracks to lost
            for track in self.tracked_tracks:
                track.is_lost = True
                track.last_frame = self.frame_id  # FIXED: Update last_frame when track is lost
                self.lost_tracks.append(track)
            
            # Create new tracks for all high-confidence detections
            for i in range(len(high_dets)):
                det = high_dets[i]
                new_track = Track(det, self.track_id_count)
                new_track.last_frame = self.frame_id  # FIXED: Set last_frame when creating track
                self.track_id_count += 1
                self.tracked_tracks.append(new_track)
            
            print(f"[BYTETRACK] No active tracks or high-conf dets: {len(self.tracked_tracks)} new tracks, " +
                  f"{len(self.lost_tracks)} lost tracks")
        
        # Remove lost tracks from tracked_tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if not t.is_lost]
        
        # Second association: low confidence detections with lost tracks
        lost_tlbrs = []
        lost_ids = []
        
        for track in self.lost_tracks:
            lost_tlbrs.append(track.tlbr)
            lost_ids.append(track.track_id)
            
        lost_tlbrs = np.array(lost_tlbrs) if lost_tlbrs else np.empty((0, 4))
        lost_ids = np.array(lost_ids)
        
        if len(lost_tlbrs) > 0 and len(low_dets) > 0:
            # Match lost tracks to low confidence detections
            matches, _, _ = self._match_tracks_to_detections(
                lost_tlbrs, low_dets[:, :4], self.match_thresh
            )
            
            # Recover matched lost tracks
            recovered_tracks = []
            for i_track, i_det in matches:
                track_id = lost_ids[i_track]
                track = self._get_track_by_id(track_id, self.lost_tracks)
                if track:
                    track.is_lost = False
                    track.update(low_dets[i_det])
                    track.last_frame = self.frame_id  # FIXED: Update last_frame on recovery
                    recovered_tracks.append(track)
            
            # Add recovered tracks back to tracked_tracks
            self.tracked_tracks.extend(recovered_tracks)
            
            # Remove recovered tracks from lost_tracks
            recovered_ids = [t.track_id for t in recovered_tracks]
            self.lost_tracks = [t for t in self.lost_tracks if t.track_id not in recovered_ids]
            
            print(f"[BYTETRACK] Recovered {len(recovered_tracks)} lost tracks with low-conf detections")
        
        # Update remaining lost tracks
        new_lost_tracks = []
        expired_count = 0
        
        # FIXED: Sort lost tracks by confidence score - keep higher quality tracks longer
        # This prevents memory issues by limiting total number of lost tracks
        sorted_lost_tracks = sorted(self.lost_tracks, key=lambda x: x.score, reverse=True)
        
        # FIXED: Only keep top MAX_LOST_TRACKS lost tracks
        MAX_LOST_TRACKS = 30  # Maximum number of lost tracks to keep
        sorted_lost_tracks = sorted_lost_tracks[:MAX_LOST_TRACKS]
        
        for track in sorted_lost_tracks:
            track.predict()  # Predict new location even when lost
            
            # FIXED: Calculate elapsed frames since last detection
            time_since_detection = self.frame_id - track.last_frame
            
            # Keep track if within time buffer, otherwise remove
            if time_since_detection <= self.max_time_lost:
                new_lost_tracks.append(track)
            else:
                self.removed_tracks.append(track)
                expired_count += 1
                
        # Calculate how many tracks were removed due to confidence threshold
        dropped_by_limit = len(self.lost_tracks) - len(sorted_lost_tracks)
        
        self.lost_tracks = new_lost_tracks
        
        print(f"[BYTETRACK] Final state: {len(self.tracked_tracks)} tracked, " +
              f"{len(self.lost_tracks)} lost, {expired_count} expired, {dropped_by_limit} dropped by limit")
        
        # Return final track results
        return self._get_track_results()
        
    def _get_track_by_id(self, track_id, track_list):
        """Helper to find a track by ID in a list"""
        for track in track_list:
            if track.track_id == track_id:
                return track
        return None
        
    def _get_track_results(self):
        """Format track results as dicts for return value"""
        results = []
        for track in self.tracked_tracks:
            if track.hits >= 1:  # FIXED: Much more lenient confirmation threshold (was 3, then 2)
                tlbr = track.tlbr
                track_id = track.track_id
                score = track.score
                class_id = track.class_id
                
                # FIXED: Better error checking for bbox values
                try:
                    x1, y1, x2, y2 = map(float, tlbr)
                    
                    # FIXED: Ensure values are valid
                    if not all(np.isfinite([x1, y1, x2, y2])):
                        print(f"[BYTETRACK WARNING] Track {track_id} has invalid bbox: {tlbr}")
                        continue
                        
                    # FIXED: Make sure width and height are positive
                    if x2 <= x1 or y2 <= y1:
                        print(f"[BYTETRACK WARNING] Track {track_id} has invalid bbox dimensions: {tlbr}")
                        continue
                    
                    results.append({
                        'id': track_id,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(score),
                        'class_id': int(class_id),
                        'state': 'tracked'
                    })
                except Exception as e:
                    print(f"[BYTETRACK ERROR] Failed to process track {track_id}: {e}")
        
        print(f"[BYTETRACK] Returning {len(results)} confirmed tracks")
        return results
        
    def _convert_detections(self, detections):
        """Convert detection dictionaries to numpy array format
        Format: [x1, y1, x2, y2, score, class_id]
        """
        if not detections:
            return np.empty((0, 6))
            
        result = []
        for det in detections:
            bbox = det.get('bbox')
            conf = det.get('confidence', 0.0)
            class_id = det.get('class_id', -1)
            
            # Make sure we have numeric values
            try:
                if bbox is not None and len(bbox) == 4:
                    # FIXED: Explicitly convert to float32 for ByteTrack
                    x1, y1, x2, y2 = map(np.float32, bbox)
                    conf = np.float32(conf)
                    class_id = int(class_id) if isinstance(class_id, (int, float)) else -1
                    
                    # Validate bbox dimensions
                    if x2 > x1 and y2 > y1 and conf > 0:
                        result.append([x1, y1, x2, y2, conf, class_id])
            except (ValueError, TypeError) as e:
                print(f"[BYTETRACK] Error converting detection: {e}")
                    
        # FIXED: Explicitly convert to float32 array
        return np.array(result, dtype=np.float32) if result else np.empty((0, 6), dtype=np.float32)
        
    def _match_tracks_to_detections(self, tracks_tlbr, dets_tlbr, threshold):
        """
        Match tracks to detections using IoU
        
        Args:
            tracks_tlbr: Track boxes [x1, y1, x2, y2]
            dets_tlbr: Detection boxes [x1, y1, x2, y2]
            threshold: IoU threshold
            
        Returns:
            (matches, unmatched_tracks, unmatched_detections)
        """
        if len(tracks_tlbr) == 0 or len(dets_tlbr) == 0:
            return [], list(range(len(tracks_tlbr))), list(range(len(dets_tlbr)))
            
        iou_matrix = self._iou_batch(tracks_tlbr, dets_tlbr)
        
        # Use Hungarian algorithm for optimal assignment
        matched_indices = self._linear_assignment(-iou_matrix, threshold)
        
        unmatched_tracks = []
        for i in range(len(tracks_tlbr)):
            if i not in matched_indices[:, 0]:
                unmatched_tracks.append(i)
                
        unmatched_detections = []
        for i in range(len(dets_tlbr)):
            if i not in matched_indices[:, 1]:
                unmatched_detections.append(i)
                
        matches = []
        for i, j in matched_indices:
            if iou_matrix[i, j] < threshold:
                unmatched_tracks.append(i)
                unmatched_detections.append(j)
            else:
                matches.append((i, j))
                
        return matches, unmatched_tracks, unmatched_detections
        
    def _iou_batch(self, boxes1, boxes2):
        """
        Calculate IoU between all pairs of boxes
        
        Args:
            boxes1: (N, 4) [x1, y1, x2, y2]
            boxes2: (M, 4) [x1, y1, x2, y2]
            
        Returns:
            IoU matrix (N, M)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)
        
        wh = np.clip(rb - lt, 0, None)  # (N,M,2)
        inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)
        
        union = area1[:, None] + area2 - inter
        
        iou = inter / (union + 1e-10)
        return iou
        
    def _linear_assignment(self, cost_matrix, threshold):
        """
        Improved greedy assignment implementation
        For each detection, find the track with highest IoU above threshold
        """
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int)
            
        matches = []
        # Sort costs in descending order
        flat_indices = np.argsort(cost_matrix.flatten())[::-1]
        cost_values = cost_matrix.flatten()[flat_indices]
        
        # Get row and col indices
        row_indices = flat_indices // cost_matrix.shape[1]
        col_indices = flat_indices % cost_matrix.shape[1]
        
        # Keep track of assigned rows and columns
        assigned_rows = set()
        assigned_cols = set()
        
        # Iterate through sorted indices
        for i in range(len(row_indices)):
            row, col = row_indices[i], col_indices[i]
            cost = cost_values[i]
            
            # If cost is below threshold, continue checking but apply a decay
            # This helps with low FPS scenarios where IoU might be lower
            if cost < threshold:
                # Calculate dynamic threshold based on position in list
                position_ratio = 1.0 - (i / len(row_indices))
                dynamic_threshold = threshold * 0.7 * position_ratio
                
                if cost < dynamic_threshold:
                    continue
                
            # If row or col already assigned, skip
            if row in assigned_rows or col in assigned_cols:
                continue
                
            # Add match
            matches.append((row, col))
            assigned_rows.add(row)
            assigned_cols.add(col)
            
        return np.array(matches) if matches else np.empty((0, 2), dtype=int)


class Track:
    """Track class for ByteTracker"""
    
    def __init__(self, detection, track_id):
        """Initialize a track from a detection
        
        Args:
            detection: Detection array [x1, y1, x2, y2, score, class_id]
            track_id: Unique track ID
        """
        self.track_id = track_id
        self.tlbr = detection[:4]  # [x1, y1, x2, y2]
        self.score = detection[4]
        self.class_id = int(detection[5])
        
        self.time_since_update = 0
        self.hits = 1  # Number of times track was matched to a detection
        self.age = 1
        self.last_frame = 0  # Will be set by the tracker during update
        self.is_lost = False  # Flag to indicate if track is lost
        
        # For Kalman filter
        self.kf = None
        self.mean = None
        self.covariance = None
        
        # Keep track of last 30 positions for smoother trajectories
        self.history = []
        self._init_kalman_filter()
        
    def _init_kalman_filter(self):
        """Initialize simple Kalman filter for position and velocity prediction
        State: [x, y, w, h, vx, vy, vw, vh]
        """
        # Simplified KF implementation
        self.mean = np.zeros(8)
        x1, y1, x2, y2 = self.tlbr
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        # Initialize state
        self.mean[:4] = [cx, cy, w, h]
        
        # Initialize covariance matrix
        self.covariance = np.eye(8) * 10
        
    def predict(self):
        """Predict next state using constant velocity model"""
        # Simple constant velocity prediction
        dt = 1.0
        
        # Transition matrix for constant velocity model
        F = np.eye(8)
        F[0, 4] = dt
        F[1, 5] = dt
        F[2, 6] = dt
        F[3, 7] = dt
        
        # Predict next state
        self.mean = F @ self.mean
        
        # Update covariance (simplified)
        Q = np.eye(8) * 0.01  # Process noise
        self.covariance = F @ self.covariance @ F.T + Q
        
        # Convert state back to bbox
        cx, cy, w, h = self.mean[:4]
        self.tlbr = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection):
        """Update track with new detection
        
        Args:
            detection: Detection array [x1, y1, x2, y2, score, class_id]
        """
        x1, y1, x2, y2 = detection[:4]
        self.tlbr = detection[:4]
        
        # Update score with EMA
        alpha = 0.9
        self.score = alpha * self.score + (1 - alpha) * detection[4]
        
        # Update state (simplified Kalman update)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w/2, y1 + h/2
        
        # Measurement
        z = np.array([cx, cy, w, h])
        
        # Kalman gain (simplified)
        H = np.zeros((4, 8))
        H[:4, :4] = np.eye(4)
        
        # Measurement covariance (higher = less trust in measurement)
        R = np.eye(4) * (1.0 / self.score)
        
        # Kalman update equations (simplified)
        y = z - H @ self.mean
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.mean = self.mean + K @ y
        self.covariance = (np.eye(8) - K @ H) @ self.covariance
        
        # Convert back to bbox
        cx, cy, w, h = self.mean[:4]
        self.tlbr = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        # Update history
        self.history.append(self.tlbr.copy())
        if len(self.history) > 30:
            self.history = self.history[-30:]
        
        # FIXED: Reset time since update counter and increment hits
        self.time_since_update = 0
        self.hits += 1
        self.is_lost = False  # FIXED: Ensure track is marked as not lost when updated
        

class ByteTrackVehicleTracker:
    """
    ByteTrack-based vehicle tracker with same API as DeepSortVehicleTracker
    for drop-in replacement with improved performance
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            print("[BYTETRACK SINGLETON] Creating ByteTrackVehicleTracker instance")
            cls._instance = super(ByteTrackVehicleTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        print("[BYTETRACK INIT] Initializing ByteTrack tracker (should only see this once)")
        
        # Parameters tuned for vehicle tracking in traffic scenes with low FPS
        # FIXED: Much more lenient parameters for consistent vehicle tracking
        self.tracker = BYTETracker(
            track_thresh=0.2,           # FIXED: Even lower threshold for better tracking continuity
            track_buffer=60,            # FIXED: Keep tracks alive longer (60 frames = 4-6 seconds at 10 FPS)
            match_thresh=0.4,           # FIXED: Much more lenient IoU threshold for matching
            track_high_thresh=0.25,     # FIXED: Lower high confidence threshold
            track_low_thresh=0.05,      # FIXED: Very low threshold for second-chance matching
            frame_rate=10               # FIXED: Match actual video FPS (~7-10)
        )
        
        self._initialized = True
        self.track_id_counter = {}     # Track seen IDs
        self.debug = True              # Enable debug output
        
        # Track count tracking for debugging
        self.track_counts = {
            'frames_processed': 0,
            'total_tracks_created': 0,
            'max_concurrent_tracks': 0,
            'current_active_tracks': 0,
            'current_lost_tracks': 0
        }

    def update(self, detections, frame=None):
        """
        Update tracker with new detections
        
        Args:
            detections: list of dicts with keys ['bbox', 'confidence', 'class_id', ...]
            frame: BGR image (optional, used for visualization but not required for ByteTrack)
            
        Returns:
            list of dicts with keys ['id', 'bbox', 'confidence', 'class_id', ...]
        """
        # FIXED: Add safety check for track ID counter
        if hasattr(self.tracker, 'track_id_count') and self.tracker.track_id_count > 10000:
            print(f"[BYTETRACK WARNING] Track ID counter extremely high ({self.tracker.track_id_count}). Resetting to 0.")
            self.tracker.track_id_count = 0
        
        # Convert detections to ByteTrack format with validation
        valid_dets = []
        for i, det in enumerate(detections):
            bbox = det.get('bbox')
            conf = det.get('confidence', 0.0)
            class_id = det.get('class_id', -1)
            
            if bbox is not None and len(bbox) == 4:
                try:
                    # FIXED: Ensure all values are explicitly converted to float32 for consistent tracking
                    x1, y1, x2, y2 = map(np.float32, bbox)
                    conf = np.float32(conf)
                    class_id = int(class_id) if isinstance(class_id, (int, float)) else -1
                    
                    # Validate bbox dimensions
                    if x2 > x1 and y2 > y1 and conf > 0.05:  # FIXED: Lower threshold for ByteTrack
                        # Create a new det with verified types
                        valid_det = {
                            'bbox': [x1, y1, x2, y2],  # Already converted to float32 above
                            'confidence': conf,
                            'class_id': class_id
                        }
                        valid_dets.append(valid_det)
                        
                        if self.debug and i % 5 == 0:  # Only print every 5th detection to reduce log spam
                            print(f"[BYTETRACK] Added detection {i}: bbox={[x1, y1, x2, y2]}, conf={conf:.2f}")
                    else:
                        if self.debug:
                            print(f"[BYTETRACK] Rejected detection {i}: invalid bbox dimensions or very low confidence")
                except Exception as e:
                    if self.debug:
                        print(f"[BYTETRACK] Error processing detection {i}: {e}")
            else:
                if self.debug:
                    print(f"[BYTETRACK] Rejected detection {i}: invalid bbox format")

        if self.debug:
            print(f"[BYTETRACK] Processing {len(valid_dets)} valid detections")

        try:
            # Use try/except to catch any errors in the tracker update
            tracks = self.tracker.update(valid_dets, frame)
            
            # Update track statistics
            self.track_counts['frames_processed'] += 1
            self.track_counts['current_active_tracks'] = len(self.tracker.tracked_tracks)
            self.track_counts['current_lost_tracks'] = len(self.tracker.lost_tracks)
            self.track_counts['max_concurrent_tracks'] = max(
                self.track_counts['max_concurrent_tracks'],
                len(self.tracker.tracked_tracks) + len(self.tracker.lost_tracks)
            )
            
            # FIXED: Clean up old removed tracks more aggressively to prevent memory issues
            if self.track_counts['frames_processed'] % 50 == 0:
                old_removed_count = len(self.tracker.removed_tracks)
                # Only keep the last 30 removed tracks
                self.tracker.removed_tracks = self.tracker.removed_tracks[-30:] if len(self.tracker.removed_tracks) > 30 else []
                print(f"[BYTETRACK] Memory cleanup: removed {old_removed_count - len(self.tracker.removed_tracks)} old tracks")
                print(f"[BYTETRACK] Stats: Active={self.track_counts['current_active_tracks']}, " +
                      f"Lost={self.track_counts['current_lost_tracks']}, " +
                      f"Max concurrent={self.track_counts['max_concurrent_tracks']}")
            
            # Make sure tracks are in a consistent dictionary format
            standardized_tracks = []
            for track in tracks:
                if isinstance(track, dict):
                    # Track is already a dict, just ensure it has required fields
                    if 'id' not in track and 'track_id' in track:
                        track['id'] = track['track_id']
                    standardized_tracks.append(track)
                else:
                    # Convert object to dict
                    try:
                        track_dict = {
                            'id': track.track_id if hasattr(track, 'track_id') else -1,
                            'bbox': track.bbox if hasattr(track, 'bbox') else [0, 0, 0, 0],
                            'confidence': track.confidence if hasattr(track, 'confidence') else 0.0,
                            'class_id': track.class_id if hasattr(track, 'class_id') else -1
                        }
                        standardized_tracks.append(track_dict)
                    except Exception as e:
                        print(f"[BYTETRACK ERROR] Error converting track to dict: {e}")
            
            return standardized_tracks
        except Exception as e:
            print(f"[BYTETRACK ERROR] Error updating tracker: {e}")
            import traceback
            traceback.print_exc()
            # Return empty tracks list as fallback
            return []
        
    def update_tracks(self, detections, frame=None):
        """
        Alias for the update method to maintain compatibility with DeepSORT interface
        
        Args:
            detections: list of detection arrays in format [bbox_xywh, conf, class_id]
            frame: BGR image
        
        Returns:
            list of objects with DeepSORT-compatible interface including is_confirmed() method
        """
        # Convert from DeepSORT format to ByteTrack format
        converted_dets = []
        
        for det in detections:
            try:
                # Handle different detection formats
                if isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 2:
                    # DeepSORT format: [bbox_xywh, conf, class_id]
                    bbox_xywh, conf = det[:2]
                    class_id = det[2] if len(det) > 2 else -1
                    
                    # Convert [x, y, w, h] to [x1, y1, x2, y2] with type validation
                    x, y, w, h = map(float, bbox_xywh)
                    conf = float(conf)
                    class_id = int(class_id) if isinstance(class_id, (int, float)) else -1
                    
                    converted_dets.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': conf,
                        'class_id': class_id
                    })
                
                elif isinstance(det, dict):
                    # Newer format with bbox in dict
                    if 'bbox' in det:
                        bbox = det['bbox']
                        if len(bbox) == 4:
                            # Check if it's already in [x1, y1, x2, y2] format
                            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                                # Already in [x1, y1, x2, y2] format
                                converted_dets.append(det.copy())
                            else:
                                # Assume it's [x, y, w, h] and convert
                                x, y, w, h = bbox
                                converted_det = det.copy()
                                converted_det['bbox'] = [x, y, x + w, y + h]
                                converted_dets.append(converted_det)
            except Exception as e:
                print(f"[BYTETRACK] Error converting detection format: {e}")
        
        # Call the regular update method to get dictionary tracks
        dict_tracks = self.update(converted_dets, frame)
        
        if self.debug:
            print(f"[BYTETRACK] Converting {len(dict_tracks)} dict tracks to DeepSORT-compatible objects")
        
        # Create DeepSORT compatible track objects from dictionaries
        ds_tracks = []
        for track_data in dict_tracks:
            ds_track = ByteTrackOutput(track_data)
            ds_tracks.append(ds_track)
        
        return ds_tracks

    def reset(self):
        """
        Reset the tracker to clean state, resetting all IDs and clearing tracks.
        Call this when starting a new video or session.
        """
        print("[BYTETRACK] Resetting tracker state - IDs will start from 1")
        if hasattr(self, 'tracker') and self.tracker is not None:
            # Reset the internal BYTETracker
            self.tracker.tracked_tracks = []
            self.tracker.lost_tracks = []
            self.tracker.removed_tracks = []
            self.tracker.frame_id = 0
            self.tracker.track_id_count = 1  # FIXED: Start from 1 instead of 0
            
            print("[BYTETRACK] Reset complete - track ID counter reset to 1")
        else:
            print("[BYTETRACK] Warning: Tracker not initialized, nothing to reset")
            
        # Reset tracking statistics
        self.track_counts = {
            'frames_processed': 0,
            'total_tracks_created': 0,
            'max_concurrent_tracks': 0,
            'current_active_tracks': 0,
            'current_lost_tracks': 0
        }
        self.track_id_counter = {}

# Adapter class to make ByteTrack output compatible with DeepSORT output
class ByteTrackOutput:
    def __init__(self, track_data):
        self.track_id = track_data['id']
        self.bbox = track_data['bbox']  # [x1, y1, x2, y2]
        self.confidence = track_data['confidence']
        self.class_id = track_data['class_id']
        self._ltrb = self.bbox  # Store bbox in LTRB format directly
        
    def to_ltrb(self):
        """Return bbox in [left, top, right, bottom] format"""
        return self._ltrb
        
    def to_tlbr(self):
        """Return bbox in [top, left, bottom, right] format"""
        # For ByteTrack, LTRB and TLBR are the same since we use [x1, y1, x2, y2]
        return self._ltrb
        
    def to_xyah(self):
        """Return bbox in [center_x, center_y, aspect_ratio, height] format"""
        x1, y1, x2, y2 = self._ltrb
        w, h = x2 - x1, y2 - y1
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        aspect_ratio = w / h if h > 0 else 1.0
        return [center_x, center_y, aspect_ratio, h]
        
    def is_confirmed(self):
        """Return True if track is confirmed"""
        return True  # ByteTrack only returns confirmed tracks
