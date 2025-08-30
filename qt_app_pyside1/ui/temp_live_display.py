from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QImage

import cv2
import numpy as np

class SimpleLiveDisplay(QWidget):
    """Simpler implementation for video display using QLabel instead of QGraphicsView"""
    
    video_dropped = Signal(str)  # For drag and drop compatibility
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create QLabel for display
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setStyleSheet("background-color: black;")
        
        # Set up drag and drop
        self.setAcceptDrops(True)
        
        # Add to layout
        self.layout.addWidget(self.display_label)
            
    def update_frame(self, pixmap):
        """Update the display with a new frame"""
        if pixmap and not pixmap.isNull():
            print(f"DEBUG: SimpleLiveDisplay updating with pixmap {pixmap.width()}x{pixmap.height()}")
            
            try:
                # Try direct approach - set the pixmap directly without scaling
                self.display_label.setPixmap(pixmap)
                
                # Force an immediate update
                self.display_label.update()
                self.repaint()  # Force a complete repaint
                print("DEBUG: SimpleLiveDisplay - pixmap set successfully")
            
            except Exception as e:
                print(f"ERROR in SimpleLiveDisplay.update_frame: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print("DEBUG: SimpleLiveDisplay received null or invalid pixmap")
            
    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # If we have a pixmap, rescale it to fit the new size
        if not self.display_label.pixmap() or self.display_label.pixmap().isNull():
            return
            
        scaled_pixmap = self.display_label.pixmap().scaled(
            self.display_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.display_label.setPixmap(scaled_pixmap)
        
    def reset_display(self):
        """Reset display to black"""
        blank = QPixmap(self.display_label.size())
        blank.fill(Qt.black)
        self.display_label.setPixmap(blank)
        
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                event.acceptProposedAction()
                
    def dropEvent(self, event):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0].toLocalFile()
            if url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                self.video_dropped.emit(url)
