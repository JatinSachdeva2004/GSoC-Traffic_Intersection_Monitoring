"""
Fallback annotation utilities for enhanced video controller.
This module provides basic implementation of the annotation functions
required by the enhanced video controller, in case the regular module fails to import.
"""

import sys
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
try:
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import Qt
    QT_AVAILABLE = True
except ImportError:
    print("⚠️ PySide6 not available, some functions will be limited")
    QT_AVAILABLE = False

# Color mapping for traffic-related classes
COLORS = {
    'person': (255, 165, 0),         # Orange
    'bicycle': (255, 0, 255),        # Magenta
    'car': (0, 255, 0),              # Green
    'motorcycle': (255, 255, 0),     # Cyan
    'bus': (0, 0, 255),              # Red
    'truck': (0, 128, 255),          # Orange-Blue
    'traffic light': (0, 165, 255),  # Orange
    'stop sign': (0, 0, 139),        # Dark Red
    'parking meter': (128, 0, 128),  # Purple
    'default': (0, 255, 255)         # Yellow as default
}

def enhanced_draw_detections(frame: np.ndarray, detections: List[Dict], 
                            show_confidence: bool = True, 
                            show_labels: bool = True) -> np.ndarray:
    """
    Draw detections on frame with enhanced visuals.
    
    Args:
        frame: Input video frame
        detections: List of detection dictionaries
        show_confidence: Whether to show confidence values
        show_labels: Whether to show class labels
        
    Returns:
        Frame with detections drawn
    """
    if not detections:
        return frame
        
    # Create a copy of the frame
    result = frame.copy()
    
    # Process each detection
    for det in detections:
        if 'bbox' not in det:
            continue
            
        # Get bounding box
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        # Get class name and confidence
        class_name = det.get('class_name', 'unknown')
        conf = det.get('confidence', 0)
        
        # Get color for this class
        color = COLORS.get(class_name.lower(), COLORS['default'])
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = ""
        if show_labels:
            label = class_name
            if show_confidence:
                label = f"{class_name} ({conf:.2f})"
            elif 'track_id' in det:
                label = f"{class_name} #{det['track_id']}"
        elif show_confidence:
            label = f"{conf:.2f}"
        elif 'track_id' in det:
            label = f"#{det['track_id']}"
            
        # Draw label if we have one
        if label:
            # Calculate label size and position
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(
                result, 
                (x1, y1), 
                (x1 + label_width, y1 - label_height - baseline - 5), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                result, 
                label, 
                (x1, y1 - baseline - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
    return result

def draw_performance_overlay(frame: np.ndarray, metrics: Dict[str, Any]) -> np.ndarray:
    """
    Draw performance metrics overlay on frame.
    
    Args:
        frame: Input video frame
        metrics: Dictionary of performance metrics
        
    Returns:
        Frame with performance overlay
    """
    if not metrics:
        return frame
        
    # Create a copy of the frame
    result = frame.copy()
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Extract metrics
    fps = metrics.get('fps', 0)
    inference_fps = metrics.get('inference_fps', 0)
    inference_time = metrics.get('inference_time', 0)
    
    # Format text
    text_lines = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_time:.1f}ms ({inference_fps:.1f} FPS)",
    ]
    
    # Draw semi-transparent background
    overlay = result.copy()
    bg_height = 30 + (len(text_lines) - 1) * 20
    cv2.rectangle(overlay, (10, 10), (250, 10 + bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
    
    # Draw text lines
    y = 30
    for text in text_lines:
        cv2.putText(
            result, 
            text, 
            (20, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1,
            cv2.LINE_AA
        )
        y += 20
        
    return result

# Qt-specific helper functions
def enhanced_cv_to_qimage(cv_img: np.ndarray) -> Optional['QImage']:
    """
    Convert OpenCV image to QImage with enhanced handling.
    
    Args:
        cv_img: OpenCV image (numpy array)
        
    Returns:
        QImage or None if conversion failed
    """
    if not QT_AVAILABLE:
        print("⚠️ Cannot convert to QImage: PySide6 not available")
        return None
        
    if cv_img is None or cv_img.size == 0:
        print("⚠️ Cannot convert empty image to QImage")
        return None
    
    try:
        height, width, channels = cv_img.shape
        
        # Ensure we're dealing with RGB or RGBA
        if channels == 3:
            # OpenCV uses BGR, we need RGB for QImage
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            format = QImage.Format_RGB888
        elif channels == 4:
            # OpenCV uses BGRA, we need RGBA for QImage
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
            format = QImage.Format_RGBA8888
        else:
            print(f"⚠️ Unsupported image format with {channels} channels")
            return None
            
        # Create QImage from numpy array
        steps = width * channels
        return QImage(cv_img.data, width, height, steps, format)
        
    except Exception as e:
        print(f"❌ Error converting image to QImage: {e}")
        return None
        
def enhanced_cv_to_pixmap(cv_img: np.ndarray) -> Optional['QPixmap']:
    """
    Convert OpenCV image to QPixmap with enhanced handling.
    
    Args:
        cv_img: OpenCV image (numpy array)
        
    Returns:
        QPixmap or None if conversion failed
    """
    if not QT_AVAILABLE:
        print("⚠️ Cannot convert to QPixmap: PySide6 not available")
        return None
        
    # Convert to QImage first
    qimg = enhanced_cv_to_qimage(cv_img)
    if qimg is None:
        return None
        
    # Convert QImage to QPixmap
    try:
        return QPixmap.fromImage(qimg)
    except Exception as e:
        print(f"❌ Error converting QImage to QPixmap: {e}")
        return None
