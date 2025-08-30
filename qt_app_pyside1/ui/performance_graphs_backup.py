"""
Real-time performance graphs for inference latency analysis
Shows when latency spikes occur with different resolutions and devices
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QGroupBox, QTabWidget, QFrame, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QLinearGradient
import numpy as np
from collections import deque
from typing import Dict, List, Any
import time

# Try to import psutil for system monitoring, use fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - system monitoring will use fallback values")

class RealTimeGraph(QWidget):
    """Custom widget for drawing real-time graphs with enhanced styling"""
    
    def __init__(self, title: str = "Graph", y_label: str = "Value", max_points: int = 300):
        super().__init__()
        self.title = title
        self.y_label = y_label
        self.max_points = max_points
        
        # Data storage
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        self.spike_markers = deque(maxlen=max_points)  # Mark spikes
        self.device_markers = deque(maxlen=max_points)  # Mark device changes
        self.resolution_markers = deque(maxlen=max_points)  # Mark resolution changes
        
        # Enhanced styling colors
        self.bg_color = QColor(18, 18, 18)  # Very dark background
        self.grid_color = QColor(40, 40, 45)  # Subtle grid
        self.line_color = QColor(0, 230, 255)  # Bright cyan
        self.spike_color = QColor(255, 77, 77)   # Bright red for spikes
        self.cpu_color = QColor(120, 180, 255)  # Light blue for CPU
        self.gpu_color = QColor(255, 165, 0)    # Orange for GPU
        self.text_color = QColor(220, 220, 220)  # Light gray text
        self.accent_color = QColor(255, 215, 0)  # Gold accent
        
        # Auto-scaling
        self.y_min = 0
        self.y_max = 100
        self.auto_scale = True
        
        # Performance counters
        self.spike_count = 0
        self.device_switches = 0
        self.resolution_changes = 0
        
        self.setMinimumSize(400, 200)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
            }
        """)
        
    def add_data_point(self, x: float, y: float, is_spike: bool = False, device: str = "CPU", is_res_change: bool = False):
        """Add a new data point to the graph"""
        self.x_data.append(x)
        self.y_data.append(y)
        self.spike_markers.append(is_spike)
        self.device_markers.append(device)
        self.resolution_markers.append(is_res_change)
        
        # Update counters
        if is_spike:
            self.spike_count += 1
        if len(self.device_markers) > 1 and device != list(self.device_markers)[-2]:
            self.device_switches += 1
        if is_res_change:
            self.resolution_changes += 1
        
        # Auto-scale Y axis with better algorithm
        if self.auto_scale and self.y_data:
            data_max = max(self.y_data)
            data_min = min(self.y_data)
            if data_max > data_min:
                padding = (data_max - data_min) * 0.15
                self.y_max = data_max + padding
                self.y_min = max(0, data_min - padding * 0.5)
            else:
                self.y_max = data_max + 10 if data_max > 0 else 100
                self.y_min = 0
        self.update()
        
    def clear_data(self):
        """Clear the graph data"""
        self.x_data.clear()
        self.y_data.clear()
        self.spike_markers.clear()
        self.device_markers.clear()
        self.resolution_markers.clear()
        self.spike_count = 0
        self.device_switches = 0
        self.resolution_changes = 0
        self.update()
        
    def add_data_point(self, x: float, y: float, is_spike: bool = False, device: str = "CPU", is_res_change: bool = False):
        """Add a new data point to the graph"""
        self.x_data.append(x)
        self.y_data.append(y)
        self.spike_markers.append(is_spike)
        self.device_markers.append(device)
        self.resolution_markers.append(is_res_change)
        
        # Auto-scale Y axis
        if self.auto_scale and self.y_data:
            data_max = max(self.y_data)
            data_min = min(self.y_data)
            padding = (data_max - data_min) * 0.1
            self.y_max = data_max + padding if data_max > 0 else 100
            self.y_min = max(0, data_min - padding)
        self.update()
        
    def clear_data(self):
        """Clear the graph data"""
        self.x_data.clear()
        self.y_data.clear()
        self.spike_markers.clear()
        self.device_markers.clear()
        self.resolution_markers.clear()
        self.update()
        
    def paintEvent(self, event):
        """Override paint event to draw the graph with enhanced styling"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        margin = 50
        graph_width = width - 2 * margin
        graph_height = height - 2 * margin
        
        # Enhanced background with subtle gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(25, 25, 30))
        gradient.setColorAt(1, QColor(15, 15, 20))
        painter.fillRect(self.rect(), QBrush(gradient))
        
        # Title with glow effect
        painter.setPen(self.accent_color)
        painter.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title_rect = painter.fontMetrics().boundingRect(self.title)
        painter.drawText(15, 25, self.title)
        
        # Enhanced axes with better styling
        painter.setPen(QPen(self.text_color, 2))
        painter.drawLine(margin, margin, margin, height - margin)  # Y-axis
        painter.drawLine(margin, height - margin, width - margin, height - margin)  # X-axis
        
        # Enhanced grid with subtle styling
        painter.setPen(QPen(self.grid_color, 1, Qt.DotLine))
        # Horizontal grid lines
        for i in range(1, 5):
            y = margin + (graph_height * i / 4)
            painter.drawLine(margin + 5, y, width - margin - 5, y)
        # Vertical grid lines  
        for i in range(1, 10):
            x = margin + (graph_width * i / 9)
            painter.drawLine(x, margin + 5, x, height - margin - 5)
        
        # Enhanced Y-axis labels with better formatting
        painter.setPen(self.text_color)
        painter.setFont(QFont("Segoe UI", 9))
        for i in range(5):
            y_val = self.y_min + (self.y_max - self.y_min) * (4 - i) / 4
            y_pos = margin + (graph_height * i / 4)
            if y_val >= 1000:
                label = f"{y_val/1000:.1f}k"
            elif y_val >= 1:
                label = f"{y_val:.1f}"
            else:
                label = f"{y_val:.2f}"
            painter.drawText(5, y_pos + 4, label)
        
        # Enhanced Y-axis label with rotation
        painter.save()
        painter.setPen(self.text_color)
        painter.setFont(QFont("Segoe UI", 10))
        painter.translate(20, height // 2)
        painter.rotate(-90)
        painter.drawText(-len(self.y_label) * 4, 0, self.y_label)
        painter.restore()
        
        # Enhanced data visualization
        if len(self.x_data) >= 2 and len(self.y_data) >= 2:
            points = []
            spike_points = []
            device_changes = []
            res_changes = []
            
            x_min = min(self.x_data) if self.x_data else 0
            x_max = max(self.x_data) if self.x_data else 1
            x_range = x_max - x_min if x_max > x_min else 1
            
            # Prepare point coordinates
            for i, (x_val, y_val, is_spike, device, is_res_change) in enumerate(zip(
                self.x_data, self.y_data, self.spike_markers, self.device_markers, self.resolution_markers
            )):
                x_screen = margin + (x_val - x_min) / x_range * graph_width
                y_screen = height - margin - (y_val - self.y_min) / (self.y_max - self.y_min) * graph_height
                points.append((x_screen, y_screen))
                
                if is_spike:
                    spike_points.append((x_screen, y_screen))
                if i > 0 and device != list(self.device_markers)[i-1]:
                    device_changes.append((x_screen, y_screen, device))
                if is_res_change:
                    res_changes.append((x_screen, y_screen))
            
            # Draw main line with enhanced styling
            if len(points) >= 2:
                painter.setPen(QPen(self.line_color, 3))
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    painter.drawLine(x1, y1, x2, y2)
                
                # Add subtle glow effect to the line
                painter.setPen(QPen(QColor(self.line_color.red(), self.line_color.green(), self.line_color.blue(), 60), 6))
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    painter.drawLine(x1, y1, x2, y2)
            
            # Enhanced spike markers
            painter.setPen(QPen(self.spike_color, 2))
            painter.setBrush(QBrush(self.spike_color))
            for x, y in spike_points:
                painter.drawEllipse(x - 4, y - 4, 8, 8)
                # Add spike indicator line
                painter.drawLine(x, y - 10, x, y + 10)
            
            # Enhanced device change indicators
            for x, y, device in device_changes:
                color = self.gpu_color if device == "GPU" else self.cpu_color
                painter.setPen(QPen(color, 3))
                painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 100)))
                painter.drawRect(x - 3, margin, 6, graph_height)
                
                # Add device label
                painter.setPen(color)
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(x - 10, margin - 5, device)
            
            # Enhanced resolution change indicators
            for x, y in res_changes:
                painter.setPen(QPen(QColor(255, 193, 7), 2))  # Amber color
                painter.drawLine(x, margin, x, height - margin)
                
                # Add resolution change marker
                painter.setBrush(QBrush(QColor(255, 193, 7)))
                painter.drawEllipse(x - 3, margin - 5, 6, 6)

class PerformanceGraphsWidget(QWidget):
    """Enhanced performance graphs widget with real-time data visualization"""
    
    # Define signals for better integration
    performance_data_updated = Signal(dict)
    spike_detected = Signal(dict)
    device_switched = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
        # Enhanced timer setup
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_graphs)
        self.system_timer = QTimer()
        self.system_timer.timeout.connect(self.update_system_metrics)
        
        try:
            self.update_timer.start(500)  # Update graphs every 500ms for smoother animation
            self.system_timer.start(1000)  # Update system metrics every second
        except Exception as e:
            print(f"❌ Error starting performance graph timers: {e}")
        
        # Enhanced data tracking
        self.start_time = time.time() if time else None
        self.latest_data = {}
        self.cpu_usage_history = deque(maxlen=300)
        self.ram_usage_history = deque(maxlen=300)  # Add missing ram_usage_history
        self.frame_counter = 0
        self.spike_threshold = 100.0  # Default spike threshold in ms
        self.previous_device = "CPU"  # Track device changes
        
        # Performance statistics
        self.latency_stats = {
            'avg': 0.0,
            'max': 0.0,
            'min': float('inf'),
            'spike_count': 0
        }
        
    def __del__(self):
        """Clean up timers when widget is destroyed"""
        try:
            if hasattr(self, 'system_timer') and self.system_timer:
                self.system_timer.stop()
                self.system_timer.deleteLater()
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.stop()
                self.update_timer.deleteLater()
        except:
            pass
    
    def closeEvent(self, event):
        """Handle widget close event"""
        try:
            if hasattr(self, 'system_timer') and self.system_timer:
                self.system_timer.stop()
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.stop()
        except:
            pass
        super().closeEvent(event)
        self.ram_usage_history = deque(maxlen=300)
        self.spike_threshold = 100  # ms threshold for latency spikes
        self.previous_device = "CPU"
        self.frame_counter = 0
        
        # Performance statistics
        self.latency_stats = {
            'avg': 0.0,
            'max': 0.0,
            'min': float('inf'),
            'spike_count': 0
        }
        
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                background: transparent;
            }
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 8px;
                margin: 2px;
            }
        """)
    
    def setup_ui(self):
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)
        
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #121212;
            }
            QScrollBar:vertical {
                background-color: #2C2C2C;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
            QScrollBar:horizontal {
                background-color: #2C2C2C;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555555;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #777777;
            }
        """)
        
        # Create scrollable content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(8)
        
        # Enhanced title section
        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        
        title_label = QLabel("🔥 Real-Time Inference Performance & Latency Spike Analysis")
        title_label.setStyleSheet("""
            font-size: 18px; 
            font-weight: bold; 
            color: #FFD700; 
            margin: 8px;
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 #FFD700, stop:1 #FFA500);
            font-weight: bold;
        """)
        title_layout.addWidget(title_label)
        
        # Enhanced system stats
        stats_layout = QHBoxLayout()
        
        self.cpu_ram_stats = QLabel("CPU: 0% | RAM: 0%")
        self.cpu_ram_stats.setStyleSheet("""
            color: #00FFFF; 
            font-weight: bold; 
            font-size: 14px; 
            margin: 4px 8px;
            padding: 4px 8px;
            background-color: rgba(0, 255, 255, 0.1);
            border-radius: 4px;
        """)
        stats_layout.addWidget(self.cpu_ram_stats)
        
        # Add current model display
        self.current_model_stats = QLabel("Model: Loading...")
        self.current_model_stats.setStyleSheet("""
            color: #FFD700; 
            font-weight: bold; 
            font-size: 14px; 
            margin: 4px 8px;
            padding: 4px 8px;
            background-color: rgba(255, 215, 0, 0.1);
            border-radius: 4px;
        """)
        stats_layout.addWidget(self.current_model_stats)
        
        title_layout.addLayout(stats_layout)
        
        title_frame.setLayout(title_layout)
        content_layout.addWidget(title_frame)
        
        # Enhanced splitter for graphs - set minimum sizes to avoid cluttering
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #333333;
                height: 3px;
            }
            QSplitter::handle:hover {
                background-color: #555555;
            }
        """)
        
        # Enhanced Latency graph
        latency_frame = QFrame()
        latency_frame.setMinimumHeight(250)  # Set minimum height to prevent cluttering
        latency_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 30, 35, 255),
                    stop:1 rgba(20, 20, 25, 255));
                border: 2px solid #00FFFF;
                border-radius: 10px;
            }
        """)
        latency_layout = QVBoxLayout(latency_frame)
        
        self.latency_graph = RealTimeGraph(
            "Inference Latency Over Time", 
            "Latency (ms)", 
            max_points=300
        )
        self.latency_graph.setParent(latency_frame)  # Explicitly set parent
        self.latency_graph.setMinimumHeight(200)  # Ensure minimum display height
        latency_layout.addWidget(self.latency_graph)
        
        latency_info = QHBoxLayout()
        self.latency_stats_label = QLabel("Avg: 0ms | Max: 0ms | Spikes: 0")
        self.latency_stats_label.setStyleSheet("""
            color: #00FFFF; 
            font-weight: bold;
            font-size: 12px;
            padding: 4px 8px;
            background-color: rgba(0, 255, 255, 0.15);
            border-radius: 4px;
            margin: 4px;
        """)
        latency_info.addWidget(self.latency_stats_label)
        latency_info.addStretch()
        latency_layout.addLayout(latency_info)
        
        latency_frame.setLayout(latency_layout)
        splitter.addWidget(latency_frame)
        
        # Enhanced FPS graph
        fps_frame = QFrame()
        fps_frame.setMinimumHeight(250)  # Set minimum height to prevent cluttering
        fps_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(30, 35, 30, 255),
                    stop:1 rgba(20, 25, 20, 255));
                border: 2px solid #00FF00;
                border-radius: 10px;
            }
        """)
        fps_layout = QVBoxLayout(fps_frame)
        
        self.fps_graph = RealTimeGraph(
            "FPS & Resolution Impact", 
            "FPS", 
            max_points=300
        )
        self.fps_graph.setParent(fps_frame)  # Explicitly set parent
        self.fps_graph.setMinimumHeight(200)  # Ensure minimum display height
        fps_layout.addWidget(self.fps_graph)
        
        fps_info = QHBoxLayout()
        self.fps_stats = QLabel("Current FPS: 0 | Resolution: - | Device: -")
        self.fps_stats.setStyleSheet("""
            color: #00FF00; 
            font-weight: bold;
            font-size: 12px;
            padding: 4px 8px;
            background-color: rgba(0, 255, 0, 0.15);
            border-radius: 4px;
            margin: 4px;
        """)
        fps_info.addWidget(self.fps_stats)
        fps_info.addStretch()
        fps_layout.addLayout(fps_info)
        
        fps_frame.setLayout(fps_layout)
        splitter.addWidget(fps_frame)
        
        # Enhanced Device switching & resolution changes graph
        device_frame = QFrame()
        device_frame.setMinimumHeight(220)  # Set minimum height to prevent cluttering
        device_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(35, 30, 30, 255),
                    stop:1 rgba(25, 20, 20, 255));
                border: 2px solid #FFB300;
                border-radius: 10px;
            }
        """)
        device_layout = QVBoxLayout(device_frame)
        
        self.device_graph = RealTimeGraph(
            "Device Switching & Resolution Changes", 
            "Events", 
            max_points=300
        )
        self.device_graph.setParent(device_frame)  # Explicitly set parent
        self.device_graph.setMinimumHeight(170)  # Ensure minimum display height
        device_layout.addWidget(self.device_graph)
        
        self.device_legend = QLabel(
            "<span style='color:#FF4444;'>●</span> CPU Spikes: 0 | "
            "<span style='color:#FFA500;'>●</span> GPU Spikes: 0 | "
            "<span style='color:#78B4FF;'>●</span> Switches: 0 | "
            "<span style='color:#FFC107;'>●</span> Res Changes: 0"
        )
        self.device_legend.setStyleSheet("""
            color: #FFB300; 
            font-size: 12px; 
            font-weight: bold; 
            margin: 4px 8px;
            padding: 4px 8px;
            background-color: rgba(255, 179, 0, 0.15);
            border-radius: 4px;
        """)
        device_layout.addWidget(self.device_legend)
        
        device_frame.setLayout(device_layout)
        splitter.addWidget(device_frame)
        
        # Set splitter proportions with minimum space for each section
        splitter.setSizes([300, 300, 250])  # Increased minimum sizes
        splitter.setChildrenCollapsible(False)  # Prevent collapsing sections
        
        content_layout.addWidget(splitter)
        content_widget.setLayout(content_layout)
        
        # Set minimum size for content widget to ensure scrolling when needed
        content_widget.setMinimumSize(400, 850)  # Minimum width and height
        
        # Add content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        fps_layout = QVBoxLayout(fps_frame)
        self.fps_graph = RealTimeGraph(
            "FPS & Resolution Impact", 
            "FPS", 
            max_points=300
        )
        fps_layout.addWidget(self.fps_graph)
        fps_info = QHBoxLayout()
        self.fps_stats = QLabel("Current FPS: 0 | Resolution: - | Device: -")
        self.fps_stats.setStyleSheet("color: #00FF00; font-weight: bold;")
        fps_info.addWidget(self.fps_stats)
        fps_info.addStretch()
        fps_layout.addLayout(fps_info)
        fps_frame.setLayout(fps_layout)
        splitter.addWidget(fps_frame)
        
        # Set splitter proportions with minimum space for each section
        splitter.setSizes([300, 300, 250])  # Increased minimum sizes
        splitter.setChildrenCollapsible(False)  # Prevent collapsing sections
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
    @Slot()
    def update_system_metrics(self):
        """Update system CPU and RAM usage"""
        try:
            # Check if the widget is still valid and not being destroyed
            if not self or not hasattr(self, 'isVisible') or not self.isVisible():
                return
                
            # Check if widgets still exist before updating
            if not hasattr(self, 'cpu_ram_stats') or not self.cpu_ram_stats:
                return
            if not hasattr(self, 'device_graph') or not self.device_graph:
                return
            
            # Check if the RealTimeGraph objects are still valid
            try:
                if hasattr(self.device_graph, 'add_data_point'):
                    # Test if the object is still valid by accessing a simple property
                    _ = self.device_graph.objectName()
                else:
                    return
            except RuntimeError:
                # Object has been deleted
                return
                
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                ram_percent = memory.percent
            else:
                # Fallback values when psutil is not available
                cpu_percent = 0.0
                ram_percent = 0.0
            
            if hasattr(self, 'cpu_usage_history'):
                self.cpu_usage_history.append(cpu_percent)
            if hasattr(self, 'ram_usage_history'):
                self.ram_usage_history.append(ram_percent)
            
            # Update display
            try:
                if PSUTIL_AVAILABLE:
                    self.cpu_ram_stats.setText(f"CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%")
                else:
                    self.cpu_ram_stats.setText("CPU: -- | RAM: -- (monitoring unavailable)")
            except RuntimeError:
                # Widget has been deleted
                return
            
            # Add CPU usage to device graph as background metric
            try:
                current_time = time.time() - self.start_time if self.start_time else 0
                self.device_graph.add_data_point(current_time, cpu_percent, device="System")
            except RuntimeError:
                # Graph has been deleted
                return
            
        except Exception as e:
            print(f"❌ Error updating system metrics: {e}")
            # Fallback in case of any error
            try:
                if hasattr(self, 'cpu_ram_stats') and self.cpu_ram_stats:
                    self.cpu_ram_stats.setText("CPU: -- | RAM: -- (error)")
            except:
                pass
    
    @Slot()
    def update_graphs(self):
        """Update graphs with latest data"""
        print(f"[PERF GRAPH DEBUG] update_graphs called, has latest_data: {bool(self.latest_data)}")
        if not self.latest_data:
            print(f"[PERF GRAPH DEBUG] No latest_data, returning")
            return
            
        # Check which graphs are still valid
        print(f"[PERF GRAPH DEBUG] Checking graph validity...")
        latency_graph_valid = False
        fps_graph_valid = False
        device_graph_valid = False
        
        if hasattr(self, 'latency_graph') and self.latency_graph:
            try:
                _ = self.latency_graph.objectName()  # Test if object is still valid
                latency_graph_valid = True
                print(f"[PERF GRAPH DEBUG] latency_graph is valid")
            except RuntimeError:
                print(f"[PERF GRAPH DEBUG] latency_graph is deleted")
                
        if hasattr(self, 'fps_graph') and self.fps_graph:
            try:
                _ = self.fps_graph.objectName()  # Test if object is still valid
                fps_graph_valid = True
                print(f"[PERF GRAPH DEBUG] fps_graph is valid")
            except RuntimeError:
                print(f"[PERF GRAPH DEBUG] fps_graph is deleted")
                
        if hasattr(self, 'device_graph') and self.device_graph:
            try:
                _ = self.device_graph.objectName()  # Test if object is still valid
                device_graph_valid = True
                print(f"[PERF GRAPH DEBUG] device_graph is valid")
            except RuntimeError:
                print(f"[PERF GRAPH DEBUG] device_graph is deleted")
                
        # If no graphs are valid, stop the timers
        if not (latency_graph_valid or fps_graph_valid or device_graph_valid):
            print(f"[PERF GRAPH DEBUG] All graphs are deleted, stopping timers and returning")
            if hasattr(self, 'update_timer') and self.update_timer:
                self.update_timer.stop()
            if hasattr(self, 'system_timer') and self.system_timer:
                self.system_timer.stop()
            return
        else:
            print(f"[PERF GRAPH DEBUG] Some graphs are still valid, continuing with updates")
            
        # Continue with graph updates for valid graphs
        try:
            chart_data = self.latest_data.get('chart_data', {})
            latency_stats = self.latest_data.get('latency_stats', {})
            current_metrics = self.latest_data.get('current_metrics', {})
            
            if not chart_data.get('timestamps'):
                return
                
            # Get the latest data point
            timestamps = chart_data.get('timestamps', [])
            if not timestamps:
                return
                
            latest_timestamp = timestamps[-1]
            current_time = time.time() - self.start_time if self.start_time else latest_timestamp
            
            # Update latency graph - only if valid
            if 'inference_latency' in chart_data and latency_graph_valid:
                try:
                    latency_values = chart_data['inference_latency']
                    if latency_values:
                        latest_latency = latency_values[-1]
                        is_spike = latest_latency > self.spike_threshold
                        device = current_metrics.get('device', 'CPU')
                        
                        self.latency_graph.add_data_point(
                            current_time, 
                            latest_latency, 
                            is_spike=is_spike,
                            device=device
                        )
                        
                        # Update latency statistics
                        self.latency_stats['max'] = max(self.latency_stats['max'], latest_latency)
                        self.latency_stats['min'] = min(self.latency_stats['min'], latest_latency)
                        if is_spike:
                            self.latency_stats['spike_count'] += 1
                            # Emit spike signal
                            self.spike_detected.emit({
                                'latency': latest_latency,
                                'timestamp': current_time,
                                'device': device
                            })
                        
                        # Calculate running average
                        if hasattr(self.latency_graph, 'y_data') and self.latency_graph.y_data:
                            self.latency_stats['avg'] = sum(self.latency_graph.y_data) / len(self.latency_graph.y_data)
                        print(f"[PERF GRAPH DEBUG] Updated latency graph: {latest_latency:.2f}ms")
                except RuntimeError:
                    # Graph object was deleted
                    print(f"[PERF GRAPH DEBUG] latency_graph was deleted during update")
                    latency_graph_valid = False
            
            # Update FPS graph - only if valid
            if 'fps' in chart_data and fps_graph_valid:
                try:
                    fps_values = chart_data['fps']
                    if fps_values:
                        latest_fps = fps_values[-1]
                        device = current_metrics.get('device', 'CPU')
                        resolution = current_metrics.get('resolution', 'Unknown')
                        
                        # Check for device switch
                        device_switched = device != self.previous_device
                        if device_switched:
                            self.device_switched.emit(device)
                            self.previous_device = device
                        
                        self.fps_graph.add_data_point(
                            current_time,
                            latest_fps,
                            device=device,
                            is_res_change=False  # Will be set by resolution change detection
                        )
                        
                        # Update FPS stats display with model name
                        model_name = current_metrics.get('model', 'Unknown')
                        self.fps_stats.setText(f"Current FPS: {latest_fps:.1f} | Resolution: {resolution} | Device: {device} | Model: {model_name}")
                        print(f"[PERF GRAPH DEBUG] Updated FPS graph: {latest_fps:.2f}")
                except RuntimeError:
                    # Graph object was deleted
                    print(f"[PERF GRAPH DEBUG] fps_graph was deleted during update")
                    fps_graph_valid = False
            
            # Update device switching graph - only if valid
            device_usage = chart_data.get('device_usage', [])
            if device_usage and device_graph_valid:
                try:
                    latest_usage = device_usage[-1]
                    device = current_metrics.get('device', 'CPU')
                    
                    self.device_graph.add_data_point(
                        current_time,
                        latest_usage * 100,  # Convert to percentage
                        device=device
                    )
                    print(f"[PERF GRAPH DEBUG] Updated device graph: {latest_usage * 100:.2f}%")
                except RuntimeError:
                    # Graph object was deleted
                    print(f"[PERF GRAPH DEBUG] device_graph was deleted during update")
                    device_graph_valid = False
                    pass
            
            # Update statistics displays
            try:
                if hasattr(self, 'latency_stats_label') and self.latency_stats_label:
                    self.latency_stats_label.setText(
                        f"Avg: {self.latency_stats['avg']:.1f}ms | "
                        f"Max: {self.latency_stats['max']:.1f}ms | "
                        f"Spikes: {self.latency_stats['spike_count']}"
                    )
            except RuntimeError:
                pass
            
            # Update device legend
            try:
                if hasattr(self, 'device_legend') and self.device_legend:
                    spike_count = getattr(self.latency_graph, 'spike_count', 0) if hasattr(self, 'latency_graph') else 0
                    gpu_spikes = getattr(self.device_graph, 'spike_count', 0) if hasattr(self, 'device_graph') else 0
                    device_switches = getattr(self.device_graph, 'device_switches', 0) if hasattr(self, 'device_graph') else 0
                    res_changes = getattr(self.device_graph, 'resolution_changes', 0) if hasattr(self, 'device_graph') else 0
                    
                    self.device_legend.setText(
                        f"<span style='color:#FF4444;'>●</span> CPU Spikes: {spike_count} | "
                        f"<span style='color:#FFA500;'>●</span> GPU Spikes: {gpu_spikes} | "
                        f"<span style='color:#78B4FF;'>●</span> Switches: {device_switches} | "
                        f"<span style='color:#FFC107;'>●</span> Res Changes: {res_changes}"
                    )
            except RuntimeError:
                pass
            
            # Update current model display
            try:
                model_name = current_metrics.get('model', 'Unknown')
                device = current_metrics.get('device', 'Unknown')
                if hasattr(self, 'current_model_stats') and self.current_model_stats:
                    self.current_model_stats.setText(f"Model: {model_name} | Device: {device}")
            except RuntimeError:
                pass
            
            self.frame_counter += 1
            
        except Exception as e:
            print(f"❌ Error updating performance graphs: {e}")
    
    def update_performance_data(self, analytics_data: Dict[str, Any]):
        """Update graphs with new analytics data, including system metrics"""
        try:
            print(f"[PERF DEBUG] update_performance_data called with: {analytics_data}")
            
            # Initialize start time if not set
            if self.start_time is None:
                self.start_time = time.time()
            
            chart_data = analytics_data.get('real_time_data', {})
            latency_stats = analytics_data.get('latency_statistics', {})
            current_metrics = analytics_data.get('current_metrics', {})
            system_metrics = analytics_data.get('system_metrics', {})
            
            if not chart_data.get('timestamps'):
                print("[PERF DEBUG] No timestamps in chart_data")
                return
            
            self.latest_data = {
                'chart_data': chart_data,
                'latency_stats': latency_stats,
                'current_metrics': current_metrics,
                'system_metrics': system_metrics
            }
            
            # Emit signal for other components
            self.performance_data_updated.emit(analytics_data)
            
            # Immediately update graphs on new data
            self.update_graphs()
            
        except Exception as e:
            print(f"❌ Error updating performance data: {e}")
    
    def clear_all_graphs(self):
        """Clear all graph data"""
        try:
            if hasattr(self, 'latency_graph') and self.latency_graph:
                try:
                    self.latency_graph.clear_data()
                except RuntimeError:
                    pass
            if hasattr(self, 'fps_graph') and self.fps_graph:
                try:
                    self.fps_graph.clear_data()
                except RuntimeError:
                    pass
            if hasattr(self, 'device_graph') and self.device_graph:
                try:
                    self.device_graph.clear_data()
                except RuntimeError:
                    pass
            
            # Reset statistics
            self.latency_stats = {
                'avg': 0.0,
                'max': 0.0,
                'min': float('inf'),
                'spike_count': 0
            }
            
            self.frame_counter = 0
            self.start_time = time.time()
            
            # Update displays with safety checks
            try:
                if hasattr(self, 'latency_stats_label') and self.latency_stats_label:
                    self.latency_stats_label.setText("Avg: 0ms | Max: 0ms | Spikes: 0")
            except RuntimeError:
                pass
            try:
                if hasattr(self, 'fps_stats') and self.fps_stats:
                    self.fps_stats.setText("Current FPS: 0 | Resolution: - | Device: -")
            except RuntimeError:
                pass
            try:
                if hasattr(self, 'device_legend') and self.device_legend:
                    self.device_legend.setText(
                        "<span style='color:#FF4444;'>●</span> CPU Spikes: 0 | "
                        "<span style='color:#FFA500;'>●</span> GPU Spikes: 0 | "
                        "<span style='color:#78B4FF;'>●</span> Switches: 0 | "
                        "<span style='color:#FFC107;'>●</span> Res Changes: 0"
                    )
            except RuntimeError:
                pass
            
        except Exception as e:
            print(f"❌ Error clearing graphs: {e}")
    
    def set_spike_threshold(self, threshold: float):
        """Set the threshold for detecting latency spikes"""
        self.spike_threshold = threshold
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics"""
        return {
            'latency_stats': self.latency_stats.copy(),
            'frame_count': self.frame_counter,
            'cpu_usage': list(self.cpu_usage_history),
            'ram_usage': list(self.ram_usage_history),
            'current_device': self.previous_device
        }
