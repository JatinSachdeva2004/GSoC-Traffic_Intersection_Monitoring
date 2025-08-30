print("✅ red_light_violation.py imported from", __file__)
print("\033[92m[DEBUG] red_light_violation.py is loaded and ready!\033[0m")

import cv2
import numpy as np
import datetime
from typing import List, Dict, Optional
from collections import defaultdict, deque
import logging
from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, get_violation_line_y
from utils.traffic_light_utils import detect_traffic_light_color

logger = logging.getLogger(__name__)

class RedLightViolationSystem:
    def __init__(self, vehicle_tracker=None, config=None):
        print("\033[92m[DEBUG] RedLightViolationSystem __init__ called!\033[0m")
        self.vehicle_tracker = vehicle_tracker
        self.config = config or {}
        self.violation_states = {}  # Track violation state per vehicle
        self.last_violation_line_y = None
        self.min_confidence = self.config.get('min_confidence', 0.5)
        self.min_violation_frames = self.config.get('min_violation_frames', 5)
        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame: np.ndarray, detections: List[Dict], 
                     traffic_light_bbox: Optional[list], frame_idx: int) -> List[Dict]:
        print(f"[REDLIGHTVIOLATION DEBUG] process_frame CALLED! detections={len(detections)} | FILE: {__file__}")
        for det in detections:
            print(f"[REDLIGHTVIOLATION DEBUG] About to check detection: {det}")
        print("\033[95m🚨 ENTERED process_frame in red_light_violation.py 🚨\033[0m")
        print(f"[DEBUG] process_frame called with frame_idx={frame_idx}, detections={len(detections)}, traffic_light_bbox={traffic_light_bbox}")
        """
        Core red light violation logic:
        - Detect crosswalk and violation line (with robust fallback)
        - Detect traffic light color from frame and bbox
        - Track vehicles by track_id
        - Report violation if vehicle crosses line while light is red and not already reported
        - Return list of violation records
        """
        # --- Violation line detection (moved here) ---
        _, _, violation_line_y, _ = detect_crosswalk_and_violation_line(frame)
        if violation_line_y is None:
            violation_line_y = int(frame.shape[0] * 0.8)
        self.last_violation_line_y = violation_line_y

        # --- Traffic light state detection ---
        traffic_light_state = 'unknown'
        if traffic_light_bbox:
            result = detect_traffic_light_color(frame, traffic_light_bbox)
            traffic_light_state = result.get('color', 'unknown')

        violations = []
        current_time = datetime.datetime.now().isoformat()

        for det in detections:
            print(f"[REDLIGHTVIOLATION DEBUG] Detection: id={det.get('id')}, class_name={det.get('class_name')}, bbox={det.get('bbox')}, conf={det.get('confidence')}")
            if not self._is_valid_vehicle(det):
                print(f"[REDLIGHTVIOLATION DEBUG] [SKIP] Not a valid vehicle: id={det.get('id')}, class_name={det.get('class_name')}, det={det}")
                continue
            track_id = det.get('id', f"temp_{frame_idx}")
            bbox = self._normalize_bbox(det['bbox'])
            vehicle_bottom = bbox[3]
            # Debug: print vehicle bottom and violation line
            print(f"[DEBUG] Vehicle id={track_id} bottom={vehicle_bottom}, violation_line_y={violation_line_y}")
            is_violating = (traffic_light_state == 'red' and 
                           vehicle_bottom > violation_line_y and
                           det.get('confidence', 0) >= self.min_confidence)
            print(f"[DEBUG] is_violating={is_violating} (traffic_light_state={traffic_light_state}, vehicle_bottom={vehicle_bottom}, violation_line_y={violation_line_y}, conf={det.get('confidence', 0)})")
            if track_id not in self.violation_states:
                self.violation_states[track_id] = {
                    'frames_violating': 0,
                    'reported': False
                }
            state = self.violation_states[track_id]
            if is_violating:
                state['frames_violating'] += 1
                print(f"[DEBUG] Vehicle id={track_id} frames_violating={state['frames_violating']}")
                if (state['frames_violating'] >= self.min_violation_frames and 
                    not state['reported']):
                    print(f"[VIOLATION] Vehicle id={track_id} triggered violation at frame {frame_idx}")
                    violations.append(self._create_violation_record(
                        det, bbox, track_id, frame_idx, current_time,
                        traffic_light_state, violation_line_y, traffic_light_bbox
                    ))
                    state['reported'] = True
            else:
                if state['frames_violating'] > 0:
                    print(f"[RESET] Vehicle id={track_id} violation state reset (was {state['frames_violating']})")
                state['frames_violating'] = 0
                state['reported'] = False

        # --- Print summary of all tracked vehicles and their violation state ---
        print("\033[94m[TRACK SUMMARY] Frame", frame_idx)
        for tid, st in self.violation_states.items():
            print(f"  id={tid}: frames_violating={st['frames_violating']}, reported={st['reported']}")
        if len(violations) == 0:
            print(f"\033[93m[NO VIOLATION] Frame {frame_idx}: No red light violation detected in this frame.\033[0m")
        print("\033[0m")

        # --- Optional: Force a violation for first 10 frames for testing ---
        # if frame_idx < 10 and detections:
        #     print("[FORCE] Forcing violation for testing!")
        #     det = detections[0]
        #     violations.append(self._create_violation_record(
        #         det, self._normalize_bbox(det['bbox']), det.get('id', 'forced'), frame_idx, current_time,
        #         traffic_light_state, violation_line_y, traffic_light_bbox
        #     ))

        return violations

    def _is_valid_vehicle(self, detection):
        valid_types = ['car', 'truck', 'bus', 'motorcycle', 'auto', 'vehicle']
        det_class = detection.get('class_name') or detection.get('class') or detection.get('label')
        if det_class is None:
            print(f"[DEBUG] No class found in detection: {detection}")
            return False
        if det_class.lower() in valid_types:
            return True
        return False

    def _normalize_bbox(self, bbox):
        if len(bbox) == 4 and (bbox[2] < 100 or bbox[3] < 100):
            x, y, w, h = bbox
            return [x, y, x + w, y + h]
        return bbox

    def _create_violation_record(self, det, bbox, track_id, frame_idx, timestamp, 
                               light_state, line_y, light_bbox):
        return {
            'type': 'RedLightViolation',
            'id': track_id,
            'details': {
                'vehicle_type': det['class_name'],
                'confidence': det.get('confidence', 0.5),
                'timestamp': timestamp,
                'bbox': bbox,
                'violation_line_y': line_y,
                'frame_no': frame_idx,
                'traffic_light_state': light_state,
                'traffic_light_bbox': light_bbox
            }
        }

def draw_violation_overlay(frame: np.ndarray, violations: List[Dict], violation_line_y: Optional[int] = None, fixed: bool = False, vehicle_tracks: Optional[dict] = None) -> np.ndarray:
    """
    Draw overlays for violations and violation line on the frame.
    - Orange for violation, green for fixed status
    - Draws violation line and bounding boxes with labels
    - Optionally draws tracked vehicle positions (magenta dots)
    """
    frame_copy = frame.copy()
    violation_color = (0, 140, 255)  # Orange
    fixed_color = (0, 200, 0)        # Green
    if violation_line_y is not None:
        cv2.line(frame_copy, (0, violation_line_y), (frame.shape[1], violation_line_y), violation_color, 3)
        cv2.putText(frame_copy, "VIOLATION LINE", (10, violation_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, violation_color, 2)
    for violation in violations:
        bbox = violation['details']['bbox']
        confidence = violation['confidence']
        vehicle_type = violation['details']['vehicle_type']
        vehicle_id = violation.get('id', None)
        x1, y1, x2, y2 = bbox
        # Always use orange for violation bboxes
        color = violation_color
        label = f"VIOLATION: {vehicle_type.upper()}"
        print(f"\033[93m[OVERLAY DRAW] Drawing violation overlay: ID={vehicle_id}, BBOX={bbox}, TYPE={vehicle_type}, CONF={confidence:.2f}\033[0m")
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame_copy, label, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame_copy, f"Confidence: {confidence:.2f}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if vehicle_id is not None:
            cv2.putText(frame_copy, f"ID: {vehicle_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    # Draw tracked positions if provided
    if vehicle_tracks is not None:
        for track_id, track in vehicle_tracks.items():
            for pos in track['positions']:
                cv2.circle(frame_copy, pos, 3, (255, 0, 255), -1)  # Magenta dots for path
    return frame_copy

# Example usage:
# system = RedLightViolationSystem()
# violations = system.process_frame(frame, detections, traffic_light_bbox, frame_idx)
# frame_with_overlay = draw_violation_overlay(frame, violations, system.last_violation_line_y)
