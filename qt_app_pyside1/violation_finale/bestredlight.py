import cv2
import numpy as np
from collections import defaultdict, deque
from qt_app_pyside.utils.crosswalk_utils2 import (
    detect_crosswalk_and_violation_line,
    get_violation_line_y,
    draw_violation_line
)
from qt_app_pyside.utils.traffic_light_utils import detect_traffic_light_color

class RedLightViolationDetector:
    def __init__(self, min_tl_conf=0.4, grace_px=5, fps=30):
        self.min_tl_conf = min_tl_conf
        self.grace_px = grace_px
        self.fps = fps
        self.vehicle_tracks = defaultdict(lambda: deque(maxlen=5))  # Track vehicle history
        self.last_violation_frame = {}  # Prevent duplicate logging

    def update_tracks(self, detections, frame_idx):
        for det in detections:
            vid = det.get('id')
            bbox = det['bbox']
            bottom_y = max(bbox[1], bbox[3])
            if vid is not None:
                self.vehicle_tracks[vid].append((frame_idx, bbox, bottom_y))

    def get_violation_line(self, frame, traffic_light_bbox=None, perspective_M=None, traffic_light_position=None):
        _, crosswalk_bbox, violation_line_y, _ = detect_crosswalk_and_violation_line(
            frame,
            traffic_light_position=traffic_light_position,
            perspective_M=perspective_M
        )
        if violation_line_y is None:
            violation_line_y = get_violation_line_y(frame, traffic_light_bbox=traffic_light_bbox, crosswalk_bbox=crosswalk_bbox)
        return violation_line_y

    def get_traffic_light_state(self, frame, traffic_light_bbox):
        return detect_traffic_light_color(frame, traffic_light_bbox)

    def detect(self, frame, detections, traffic_light_bbox, frame_idx):
        annotated = frame.copy()
        violations = []

        # Detect traffic light state
        tl_info = self.get_traffic_light_state(frame, traffic_light_bbox)
        tl_color = tl_info.get('color', 'unknown')
        tl_conf = tl_info.get('confidence', 0.0)

        # Detect violation line
        violation_line_y = self.get_violation_line(frame, traffic_light_bbox)

        # Draw violation line
        if violation_line_y is not None:
            annotated = draw_violation_line(annotated, violation_line_y, color=(0, 255, 255), thickness=4, label="Violation Line")

        # If light is not red or confidence is low, return frame
        if tl_color != 'red' or tl_conf < self.min_tl_conf or violation_line_y is None:
            return annotated, []

        # Update vehicle tracks
        self.update_tracks(detections, frame_idx)

        for det in detections:
            vid = det.get('id')
            bbox = det['bbox']
            bottom_y = max(bbox[1], bbox[3])

            # Check if vehicle has crossed the violation line (with grace)
            if bottom_y < violation_line_y + self.grace_px:
                continue

            # Avoid duplicate logging within a short frame window
            if vid in self.last_violation_frame and frame_idx - self.last_violation_frame[vid] < 15:
                continue

            # Draw violation indication
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"VIOLATION"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if vid is not None:
                cv2.putText(annotated, f"ID:{vid}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Log violation
            violations.append({
                "vehicle_id": vid,
                "frame_idx": frame_idx,
                "bbox": bbox,
                "traffic_light_color": tl_color,
                "traffic_light_confidence": tl_conf,
                "violation_line_y": violation_line_y
            })

            self.last_violation_frame[vid] = frame_idx

        return annotated, violations
