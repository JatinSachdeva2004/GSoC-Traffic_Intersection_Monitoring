# Utility for drawing detections, tracks, and violations on frames
import utils
from red_light_violation_pipeline import RedLightViolationPipeline
import numpy as np
from PySide6.QtGui import QPixmap
from .annotation_utils import resize_frame_for_display, convert_cv_to_pixmap

def enhanced_annotate_frame(app, frame, detections, violations):
    import cv2
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return np.zeros((300, 300, 3), dtype=np.uint8)
    annotated_frame = frame.copy()
    if detections is None:
        detections = []
    if violations is None:
        violations = []
    if len(detections) > 0:
        if hasattr(app, 'tracker') and app.tracker:
            try:
                ds_dets = []
                for det in detections:
                    if 'bbox' not in det:
                        continue
                    try:
                        bbox = det['bbox']
                        if len(bbox) < 4:
                            continue
                        x1, y1, x2, y2 = bbox
                        w = x2 - x1
                        h = y2 - y1
                        if w <= 0 or h <= 0:
                            continue
                        conf = det.get('confidence', 0.0)
                        class_name = det.get('class_name', 'unknown')
                        ds_dets.append(([x1, y1, w, h], conf, class_name))
                    except Exception:
                        continue
                if ds_dets:
                    tracks = app.tracker.update_tracks(ds_dets, frame=frame.copy())
                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        tid = track.track_id
                        ltrb = track.to_ltrb()
                        for det in detections:
                            if 'bbox' not in det:
                                continue
                            try:
                                bbox = det['bbox']
                                if len(bbox) < 4:
                                    continue
                                dx1, dy1, dx2, dy2 = bbox
                                iou = utils.bbox_iou((dx1, dy1, dx2, dy2), tuple(map(int, ltrb)))
                                if iou > 0.5:
                                    det['track_id'] = tid
                                    break
                            except Exception:
                                continue
            except Exception:
                pass
    # IMPORTANT: All OpenCV drawing (including violation line) must be done on BGR frame before converting to RGB/QImage/QPixmap.
    # Example usage in pipeline:
    # 1. Draw violation line and all overlays on annotated_frame (BGR)
    # 2. Resize for display: display_frame = resize_frame_for_display(annotated_frame, ...)
    # 3. Convert to QPixmap: pixmap = convert_cv_to_pixmap(display_frame) or enhanced_cv_to_pixmap(display_frame)
    # Do NOT convert to RGB before drawing overlays!
    try:
        show_labels = app.config.get('display', {}).get('show_labels', True)
        show_confidence = app.config.get('display', {}).get('show_confidence', True)
        annotated_frame = utils.draw_detections(annotated_frame, detections, show_labels, show_confidence)
        annotated_frame = utils.draw_violations(annotated_frame, violations)
        return annotated_frame
    except Exception:
        return frame.copy()

# def pipeline_with_violation_line(frame: np.ndarray, draw_violation_line_func, violation_line_y: int = None) -> QPixmap:
#     """
#     Example pipeline to ensure violation line is drawn and color order is correct.
#     Args:
#         frame: Input BGR frame (np.ndarray)
#         draw_violation_line_func: Function to draw violation line (should accept BGR frame)
#         violation_line_y: Y position for the violation line (int)
#     Returns:
#         QPixmap ready for display
#     """
#     # 1. Draw violation line and overlays on BGR frame
#     annotated_frame = frame.copy()
#     if violation_line_y is not None:
#         annotated_frame = draw_violation_line_func(annotated_frame, violation_line_y, color=(0, 0, 255), label='VIOLATION LINE')
#     # 2. Resize for display
#     display_frame = resize_frame_for_display(annotated_frame, max_width=1280, max_height=720)
#     # 3. Convert to QPixmap (handles BGR->RGB)
#     pixmap = convert_cv_to_pixmap(display_frame)
#     return pixmap
