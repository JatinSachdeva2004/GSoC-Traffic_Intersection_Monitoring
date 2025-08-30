"""
Live View - Real-time detection and monitoring
Connects to existing video controller and live detection logic.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QComboBox, QSlider, QSpinBox, QGroupBox,
    QGridLayout, QFrame, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSize
from PySide6.QtGui import QPixmap, QPainter, QBrush, QColor, QFont

import cv2
import numpy as np
from pathlib import Path

# Import finale components
from ..styles import FinaleStyles, MaterialColors
from ..icons import FinaleIcons

class VideoDisplayWidget(QLabel):
    """
    Advanced video display widget with overlays and interactions.
    """
    
    frame_clicked = Signal(int, int)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #424242;
                border-radius: 8px;
                background-color: #1a1a1a;
            }
        """)
        
        # State
        self.current_pixmap = None
        self.overlay_enabled = True
        
        # Default placeholder
        self.set_placeholder()
    
    def set_placeholder(self):
        """Set placeholder image when no video is loaded"""
        placeholder = QPixmap(640, 480)
        placeholder.fill(QColor(26, 26, 26))
        
        painter = QPainter(placeholder)
        painter.setPen(QColor(117, 117, 117))
        painter.setFont(QFont("Segoe UI", 16))
        painter.drawText(placeholder.rect(), Qt.AlignCenter, "No Video Source\nClick to select a file")
        painter.end()
        
        self.setPixmap(placeholder)
    
    def update_frame(self, pixmap, detections=None):
        """Update frame with detections overlay"""
        if pixmap is None:
            return
            
        self.current_pixmap = pixmap
        
        if self.overlay_enabled and detections:
            # Draw detection overlays
            pixmap = self.add_detection_overlay(pixmap, detections)
        
        self.setPixmap(pixmap)
    
    def add_detection_overlay(self, pixmap, detections):
        """Add detection overlays to pixmap"""
        if not detections:
            return pixmap
            
        # Create a copy to draw on
        overlay_pixmap = QPixmap(pixmap)
        painter = QPainter(overlay_pixmap)
        
        # Draw detection boxes
        for detection in detections:
            # Extract detection info (format depends on backend)
            if isinstance(detection, dict):
                bbox = detection.get('bbox', [])
                confidence = detection.get('confidence', 0.0)
                class_name = detection.get('class', 'unknown')
            else:
                # Handle other detection formats
                continue
                
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                
                # Draw bounding box
                painter.setPen(QColor(MaterialColors.primary))
                painter.drawRect(int(x1), int(y1), int(x2-x1), int(y2-y1))
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                painter.setPen(QColor(MaterialColors.text_primary))
                painter.drawText(int(x1), int(y1-5), label)
        
        painter.end()
        return overlay_pixmap
    
    def mousePressEvent(self, event):
        """Handle mouse click events"""
        if event.button() == Qt.LeftButton:
            self.frame_clicked.emit(event.x(), event.y())
        super().mousePressEvent(event)

class SourceControlWidget(QGroupBox):
    """
    Widget for controlling video source (file, camera, stream).
    """
    
    source_changed = Signal(str)  # source path/url
    
    def __init__(self, parent=None):
        super().__init__("Video Source", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the source control UI"""
        layout = QVBoxLayout(self)
        
        # Source type selection
        source_layout = QHBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Select Source", "Video File", "Camera", "RTSP Stream"])
        self.source_combo.currentTextChanged.connect(self.on_source_type_changed)
        
        self.browse_btn = QPushButton(FinaleIcons.get_icon("folder"), "Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        self.browse_btn.setEnabled(False)
        
        source_layout.addWidget(QLabel("Type:"))
        source_layout.addWidget(self.source_combo)
        source_layout.addWidget(self.browse_btn)
        
        layout.addLayout(source_layout)
        
        # Source path/URL input
        path_layout = QHBoxLayout()
        
        self.path_label = QLabel("Path/URL:")
        self.path_display = QLabel("No source selected")
        self.path_display.setStyleSheet("QLabel { color: #757575; font-style: italic; }")
        
        path_layout.addWidget(self.path_label)
        path_layout.addWidget(self.path_display, 1)
        
        layout.addLayout(path_layout)
        
        # Camera settings (initially hidden)
        self.camera_widget = QWidget()
        camera_layout = QHBoxLayout(self.camera_widget)
        
        camera_layout.addWidget(QLabel("Camera ID:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        camera_layout.addWidget(self.camera_spin)
        
        camera_layout.addStretch()
        self.camera_widget.hide()
        
        layout.addWidget(self.camera_widget)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())
    
    @Slot(str)
    def on_source_type_changed(self, source_type):
        """Handle source type change"""
        if source_type == "Video File":
            self.browse_btn.setEnabled(True)
            self.camera_widget.hide()
        elif source_type == "Camera":
            self.browse_btn.setEnabled(False)
            self.camera_widget.show()
            self.path_display.setText(f"Camera {self.camera_spin.value()}")
            self.source_changed.emit(str(self.camera_spin.value()))
        elif source_type == "RTSP Stream":
            self.browse_btn.setEnabled(False)
            self.camera_widget.hide()
            # Could add RTSP URL input here
        else:
            self.browse_btn.setEnabled(False)
            self.camera_widget.hide()
    
    @Slot()
    def browse_file(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.path_display.setText(file_path)
            self.source_changed.emit(file_path)

class DetectionControlWidget(QGroupBox):
    """
    Widget for controlling detection parameters.
    """
    
    confidence_changed = Signal(float)
    nms_threshold_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__("Detection Settings", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup detection control UI"""
        layout = QGridLayout(self)
        
        # Confidence threshold
        layout.addWidget(QLabel("Confidence:"), 0, 0)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(30)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        
        self.confidence_label = QLabel("0.30")
        self.confidence_label.setMinimumWidth(40)
        
        layout.addWidget(self.confidence_slider, 0, 1)
        layout.addWidget(self.confidence_label, 0, 2)
        
        # NMS threshold
        layout.addWidget(QLabel("NMS Threshold:"), 1, 0)
        
        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(1, 100)
        self.nms_slider.setValue(45)
        self.nms_slider.valueChanged.connect(self.on_nms_changed)
        
        self.nms_label = QLabel("0.45")
        self.nms_label.setMinimumWidth(40)
        
        layout.addWidget(self.nms_slider, 1, 1)
        layout.addWidget(self.nms_label, 1, 2)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())
    
    @Slot(int)
    def on_confidence_changed(self, value):
        """Handle confidence threshold change"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        self.confidence_changed.emit(confidence)
    
    @Slot(int)
    def on_nms_changed(self, value):
        """Handle NMS threshold change"""
        nms = value / 100.0
        self.nms_label.setText(f"{nms:.2f}")
        self.nms_threshold_changed.emit(nms)

class LiveView(QWidget):
    """
    Main live detection view.
    Displays real-time video with detection overlays and controls.
    """
    
    source_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_detections = []
        
    def setup_ui(self):
        """Setup the live view UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Main video display area
        video_layout = QVBoxLayout()
        
        self.video_widget = VideoDisplayWidget()
        self.video_widget.frame_clicked.connect(self.on_frame_clicked)
        
        video_layout.addWidget(self.video_widget, 1)
        
        # Video controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton(FinaleIcons.get_icon("play"), "")
        self.play_btn.setToolTip("Play/Pause")
        self.play_btn.setFixedSize(40, 40)
        
        self.stop_btn = QPushButton(FinaleIcons.get_icon("stop"), "")
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedSize(40, 40)
        
        self.record_btn = QPushButton(FinaleIcons.get_icon("record"), "")
        self.record_btn.setToolTip("Record")
        self.record_btn.setFixedSize(40, 40)
        self.record_btn.setCheckable(True)
        
        self.snapshot_btn = QPushButton(FinaleIcons.get_icon("camera"), "")
        self.snapshot_btn.setToolTip("Take Snapshot")
        self.snapshot_btn.setFixedSize(40, 40)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.record_btn)
        controls_layout.addWidget(self.snapshot_btn)
        controls_layout.addStretch()
        
        # Overlay toggle
        self.overlay_btn = QPushButton(FinaleIcons.get_icon("visibility"), "Overlays")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setChecked(True)
        self.overlay_btn.toggled.connect(self.toggle_overlays)
        
        controls_layout.addWidget(self.overlay_btn)
        
        video_layout.addLayout(controls_layout)
        layout.addLayout(video_layout, 3)
        
        # Right panel for controls
        right_panel = QVBoxLayout()
        
        # Source control
        self.source_control = SourceControlWidget()
        self.source_control.source_changed.connect(self.source_changed.emit)
        right_panel.addWidget(self.source_control)
        
        # Detection control
        self.detection_control = DetectionControlWidget()
        right_panel.addWidget(self.detection_control)
        
        # Detection info
        self.info_widget = QGroupBox("Detection Info")
        info_layout = QVBoxLayout(self.info_widget)
        
        self.detection_count_label = QLabel("Detections: 0")
        self.fps_label = QLabel("FPS: 0.0")
        self.resolution_label = QLabel("Resolution: N/A")
        
        info_layout.addWidget(self.detection_count_label)
        info_layout.addWidget(self.fps_label)
        info_layout.addWidget(self.resolution_label)
        
        self.info_widget.setStyleSheet(FinaleStyles.get_group_box_style())
        right_panel.addWidget(self.info_widget)
        
        right_panel.addStretch()
        
        layout.addLayout(right_panel, 1)
        
        # Apply theme
        self.apply_theme(True)
    
    def update_frame(self, pixmap, detections=None):
        """Update the video frame with detections"""
        if pixmap is None:
            return
            
        self.current_detections = detections or []
        self.video_widget.update_frame(pixmap, self.current_detections)
        
        # Update detection info
        self.detection_count_label.setText(f"Detections: {len(self.current_detections)}")
        
        if pixmap:
            size = pixmap.size()
            self.resolution_label.setText(f"Resolution: {size.width()}x{size.height()}")
    
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @Slot(bool)
    def toggle_overlays(self, enabled):
        """Toggle detection overlays"""
        self.video_widget.overlay_enabled = enabled
        # Refresh current frame
        if self.video_widget.current_pixmap:
            self.video_widget.update_frame(self.video_widget.current_pixmap, self.current_detections)
    
    @Slot(int, int)
    def on_frame_clicked(self, x, y):
        """Handle frame click for interaction"""
        print(f"Frame clicked at ({x}, {y})")
        # Could be used for region selection, etc.
    
    def apply_theme(self, dark_mode=True):
        """Apply theme to the view"""
        if dark_mode:
            self.setStyleSheet(f"""
                QWidget {{
                    background-color: {MaterialColors.surface};
                    color: {MaterialColors.text_primary};
                }}
                QPushButton {{
                    background-color: {MaterialColors.primary};
                    color: {MaterialColors.text_on_primary};
                    border: none;
                    border-radius: 20px;
                    padding: 8px;
                }}
                QPushButton:hover {{
                    background-color: {MaterialColors.primary_variant};
                }}
                QPushButton:checked {{
                    background-color: {MaterialColors.secondary};
                }}
            """)
