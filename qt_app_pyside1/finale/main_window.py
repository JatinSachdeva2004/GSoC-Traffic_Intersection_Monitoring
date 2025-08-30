"""
Finale UI - Modern Main Window
Advanced traffic monitoring interface with Material Design and dark theme.
Connects to existing detection/violation logic from qt_app_pyside.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QDockWidget, QSplitter, QFrame, QMessageBox, QApplication,
    QFileDialog, QStatusBar, QMenuBar, QMenu, QToolBar
)
from PySide6.QtCore import Qt, QTimer, QSettings, QSize, Signal, Slot, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QPixmap, QAction, QPainter, QBrush, QColor

import os
import sys
import json
import time
import traceback
from pathlib import Path

# Import finale UI components
try:
    # Try relative imports first (when running as a package)
    from .styles import FinaleStyles, MaterialColors
    from .icons import FinaleIcons
    from .toolbar import FinaleToolbar
    from .components.stats_widgets import StatsWidget, MetricsWidget, SystemResourceWidget
    from .views import LiveView, AnalyticsView, ViolationsView, SettingsView
except ImportError:
    # Fallback to direct imports (when running as script)
    try:
        from styles import FinaleStyles, MaterialColors
        from icons import FinaleIcons
        from toolbar import FinaleToolbar
        from components.stats_widgets import StatsWidget, MetricsWidget, SystemResourceWidget
        from views import LiveView, AnalyticsView, ViolationsView, SettingsView
    except ImportError:
        print('Error importing main window components')

# Import existing detection/violation logic from qt_app_pyside
sys.path.append(str(Path(__file__).parent.parent))
try:
    from controllers.model_manager import ModelManager
    from controllers.video_controller_new import VideoController
    from controllers.analytics_controller import AnalyticsController
    from controllers.performance_overlay import PerformanceOverlay
    # Import detection_openvino for advanced detection logic
    from detection_openvino import OpenVINOVehicleDetector
    from red_light_violation_pipeline import RedLightViolationPipeline
    from utils.helpers import load_configuration, save_configuration
    from utils.annotation_utils import draw_detections, convert_cv_to_pixmap
    from utils.enhanced_annotation_utils import enhanced_draw_detections
    from utils.traffic_light_utils import detect_traffic_light_color
except ImportError as e:
    print(f"Warning: Could not import some dependencies: {e}")
    # Fallback imports
    from controllers.model_manager import ModelManager
    VideoController = None
    def load_configuration(path): return {}
    def save_configuration(config, path): pass

class FinaleMainWindow(QMainWindow):
    """
    Modern main window for traffic monitoring with advanced UI.
    Connects to existing detection/violation logic without modifying it.
    """
    
    # Signals for UI updates
    theme_changed = Signal(bool)  # dark_mode
    view_changed = Signal(str)    # view_name
    fullscreen_toggled = Signal(bool)
    
    def __init__(self):
        super().__init__()
        
        # Initialize settings and configuration
        self.settings = QSettings("Finale", "TrafficMonitoring")
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qt_app_pyside", "config.json")
        self.config = load_configuration(self.config_file)
        
        # UI state
        self.dark_mode = True
        self.current_view = "live"
        self.is_fullscreen = False
        
        # Animation system
        self.animations = {}
        
        # Initialize UI
        self.setup_ui()
        
        # Initialize backend controllers (existing logic)
        self.setup_controllers()
        
        # Connect signals
        self.connect_signals()
        
        # Apply theme and restore settings
        self.apply_theme()
        self.restore_settings()
        
        # Show ready message
        self.statusBar().showMessage("Finale UI Ready", 3000)
        
    def setup_ui(self):
        """Set up the modern user interface"""
        # Window properties with advanced styling
        self.setWindowTitle("Finale Traffic Monitoring System")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Set window icon
        self.setWindowIcon(FinaleIcons.get_icon("traffic_monitoring"))
        
        # Create central widget with modern layout
        self.setup_central_widget()
        
        # Create modern toolbar
        self.setup_toolbar()
        
        # Create docked widgets
        self.setup_dock_widgets()
        
        # Create status bar
        self.setup_status_bar()
        
        # Create menu bar
        self.setup_menu_bar()
        
        # Apply initial styling
        self.setStyleSheet(FinaleStyles.get_main_window_style())
        
    def setup_central_widget(self):
        """Create the central widget with modern tabbed interface"""
        # Create main splitter for flexible layout
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Create left panel for main content
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        
        # Create modern tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setMovable(True)
        self.tabs.setTabsClosable(False)
        
        # Create views (these will be implemented next)
        self.live_view = LiveView()
        self.analytics_view = AnalyticsView()
        self.violations_view = ViolationsView()
        self.settings_view = SettingsView()
        
        # Add tabs with icons
        self.tabs.addTab(self.live_view, FinaleIcons.get_icon("live"), "Live Detection")
        self.tabs.addTab(self.analytics_view, FinaleIcons.get_icon("analytics"), "Analytics")
        self.tabs.addTab(self.violations_view, FinaleIcons.get_icon("warning"), "Violations")
        self.tabs.addTab(self.settings_view, FinaleIcons.get_icon("settings"), "Settings")
        
        # Style the tab widget
        self.tabs.setStyleSheet(FinaleStyles.get_tab_widget_style())
        
        # Add to layout
        self.content_layout.addWidget(self.tabs)
        self.main_splitter.addWidget(self.content_widget)
        
        # Set as central widget
        self.setCentralWidget(self.main_splitter)
        
    def setup_toolbar(self):
        """Create the modern toolbar"""
        self.toolbar = FinaleToolbar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)
        
        # Connect toolbar signals
        self.toolbar.play_clicked.connect(self.on_play_clicked)
        self.toolbar.pause_clicked.connect(self.on_pause_clicked)
        self.toolbar.stop_clicked.connect(self.on_stop_clicked)
        self.toolbar.record_clicked.connect(self.on_record_clicked)
        self.toolbar.snapshot_clicked.connect(self.on_snapshot_clicked)
        self.toolbar.settings_clicked.connect(self.show_settings)
        self.toolbar.fullscreen_clicked.connect(self.toggle_fullscreen)
        self.toolbar.theme_changed.connect(self.set_dark_mode)
        
    def setup_dock_widgets(self):
        """Create docked widgets for statistics and controls"""
        # Stats dock widget
        self.stats_dock = QDockWidget("Statistics", self)
        self.stats_dock.setObjectName("StatsDock")
        self.stats_widget = StatsWidget()
        self.stats_dock.setWidget(self.stats_widget)
        self.stats_dock.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        
        # Metrics dock widget
        self.metrics_dock = QDockWidget("Performance", self)
        self.metrics_dock.setObjectName("MetricsDock")
        self.metrics_widget = MetricsWidget()
        self.metrics_dock.setWidget(self.metrics_widget)
        self.metrics_dock.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.metrics_dock)
        
        # System resources dock widget
        self.system_dock = QDockWidget("System", self)
        self.system_dock.setObjectName("SystemDock")
        self.system_widget = SystemResourceWidget()
        self.system_dock.setWidget(self.system_widget)
        self.system_dock.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetClosable |
            QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, self.system_dock)
        
        # Tabify dock widgets for space efficiency
        self.tabifyDockWidget(self.stats_dock, self.metrics_dock)
        self.tabifyDockWidget(self.metrics_dock, self.system_dock)
        
        # Show stats dock by default
        self.stats_dock.raise_()
        
        # Apply dock widget styling
        for dock in [self.stats_dock, self.metrics_dock, self.system_dock]:
            dock.setStyleSheet(FinaleStyles.get_dock_widget_style())
        
    def setup_status_bar(self):
        """Create modern status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets to status bar
        self.fps_label = QWidget()
        self.connection_label = QWidget()
        self.model_label = QWidget()
        
        self.status_bar.addPermanentWidget(self.fps_label)
        self.status_bar.addPermanentWidget(self.connection_label)
        self.status_bar.addPermanentWidget(self.model_label)
        
        # Style status bar
        self.status_bar.setStyleSheet(FinaleStyles.get_status_bar_style())
        
    def setup_menu_bar(self):
        """Create modern menu bar"""
        self.menu_bar = self.menuBar()
        
        # File menu
        file_menu = self.menu_bar.addMenu("&File")
        
        open_action = QAction(FinaleIcons.get_icon("folder"), "&Open Video", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction(FinaleIcons.get_icon("save"), "&Save Config", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction(FinaleIcons.get_icon("exit"), "E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menu_bar.addMenu("&View")
        
        fullscreen_action = QAction(FinaleIcons.get_icon("fullscreen"), "&Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        theme_action = QAction(FinaleIcons.get_icon("theme"), "&Dark Theme", self)
        theme_action.setCheckable(True)
        theme_action.setChecked(self.dark_mode)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        # Tools menu
        tools_menu = self.menu_bar.addMenu("&Tools")
        
        settings_action = QAction(FinaleIcons.get_icon("settings"), "&Settings", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = self.menu_bar.addMenu("&Help")
        
        about_action = QAction(FinaleIcons.get_icon("info"), "&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # Style menu bar
        self.menu_bar.setStyleSheet(FinaleStyles.get_menu_bar_style())
        
    def setup_controllers(self):
        """Initialize backend controllers (existing logic)"""
        try:
            # Initialize model manager (existing from qt_app_pyside)
            self.model_manager = ModelManager(self.config_file)
            
            # Initialize video controller (existing from qt_app_pyside)
            self.video_controller = VideoController(self.model_manager)
            
            # Initialize analytics controller (existing from qt_app_pyside)
            self.analytics_controller = AnalyticsController()
            
            # Initialize performance overlay (existing from qt_app_pyside)
            self.performance_overlay = PerformanceOverlay()
            
            print("✅ Backend controllers initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing controllers: {e}")
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize backend controllers:\n{str(e)}")
        
    def connect_signals(self):
        """Connect signals between UI and backend"""
        try:
            # Connect video controller signals to UI updates
            if hasattr(self.video_controller, 'frame_ready'):
                self.video_controller.frame_ready.connect(self.on_frame_ready)
            
            if hasattr(self.video_controller, 'stats_ready'):
                self.video_controller.stats_ready.connect(self.on_stats_ready)
            
            if hasattr(self.video_controller, 'violation_detected'):
                self.video_controller.violation_detected.connect(self.on_violation_detected)
            
            # Connect tab change signal
            self.tabs.currentChanged.connect(self.on_tab_changed)
            
            # Connect view signals to backend
            self.live_view.source_changed.connect(self.on_source_changed)
            
            print("✅ Signals connected successfully")
            
        except Exception as e:
            print(f"❌ Error connecting signals: {e}")
    
    # Event handlers for UI interactions
    @Slot()
    def on_play_clicked(self):
        """Handle play button click"""
        if hasattr(self.video_controller, 'start'):
            self.video_controller.start()
        self.toolbar.set_playback_state("playing")
        
    @Slot()
    def on_pause_clicked(self):
        """Handle pause button click"""
        if hasattr(self.video_controller, 'pause'):
            self.video_controller.pause()
        self.toolbar.set_playback_state("paused")
        
    @Slot()
    def on_stop_clicked(self):
        """Handle stop button click"""
        if hasattr(self.video_controller, 'stop'):
            self.video_controller.stop()
        self.toolbar.set_playback_state("stopped")
        
    @Slot()
    def on_record_clicked(self):
        """Handle record button click"""
        # Implementation depends on existing recording logic
        pass
        
    @Slot()
    def on_snapshot_clicked(self):
        """Handle snapshot button click"""
        # Implementation depends on existing snapshot logic
        pass
    
    # Backend signal handlers
    @Slot(object, object, dict)
    def on_frame_ready(self, pixmap, detections, metrics):
        """Handle frame ready signal from video controller"""
        # Update live view
        if self.current_view == "live":
            self.live_view.update_frame(pixmap, detections)
        
        # Update toolbar status
        self.toolbar.update_status("processing", True)
        
    @Slot(dict)
    def on_stats_ready(self, stats):
        """Handle stats ready signal from video controller"""
        # Update stats widgets
        self.stats_widget.update_stats(stats)
        self.metrics_widget.update_metrics(stats)
        
        # Update toolbar FPS
        if 'fps' in stats:
            self.toolbar.update_fps(stats['fps'])
    
    @Slot(dict)
    def on_violation_detected(self, violation_data):
        """Handle violation detected signal"""
        # Update violations view
        self.violations_view.add_violation(violation_data)
        
        # Update toolbar status
        self.toolbar.update_status("violation", True)
        
        # Play notification sound/animation if enabled
        self.play_violation_notification()
    
    @Slot(str)
    def on_source_changed(self, source_path):
        """Handle source change from live view"""
        if hasattr(self.video_controller, 'set_source'):
            self.video_controller.set_source(source_path)
        
    @Slot(int)
    def on_tab_changed(self, index):
        """Handle tab change"""
        tab_names = ["live", "analytics", "violations", "settings"]
        if 0 <= index < len(tab_names):
            self.current_view = tab_names[index]
            self.view_changed.emit(self.current_view)
    
    # UI control methods
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
            self.is_fullscreen = False
        else:
            self.showFullScreen()
            self.is_fullscreen = True
        
        self.fullscreen_toggled.emit(self.is_fullscreen)
    
    def toggle_theme(self):
        """Toggle between dark and light theme"""
        self.set_dark_mode(not self.dark_mode)
    
    def set_dark_mode(self, dark_mode):
        """Set theme mode"""
        self.dark_mode = dark_mode
        self.apply_theme()
        self.theme_changed.emit(self.dark_mode)
    
    def apply_theme(self):
        """Apply current theme to all UI elements"""
        # Apply main styles
        self.setStyleSheet(FinaleStyles.get_main_window_style(self.dark_mode))
        
        # Update all child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'apply_theme'):
                child.apply_theme(self.dark_mode)
        
        # Update color scheme
        if self.dark_mode:
            MaterialColors.apply_dark_theme()
        else:
            MaterialColors.apply_light_theme()
    
    def show_settings(self):
        """Show settings view"""
        self.tabs.setCurrentWidget(self.settings_view)
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Finale UI", 
                         "Finale Traffic Monitoring System\n"
                         "Modern UI for OpenVINO-based traffic detection\n"
                         "Built with PySide6 and Material Design")
    
    def open_file(self):
        """Open file dialog for video source"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if file_path:
            self.on_source_changed(file_path)
    
    def save_config(self):
        """Save current configuration"""
        try:
            save_configuration(self.config, self.config_file)
            self.statusBar().showMessage("Configuration saved", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save configuration:\n{str(e)}")
    
    def play_violation_notification(self):
        """Play violation notification (visual/audio)"""
        # Create a brief red flash animation
        self.create_violation_flash()
    
    def create_violation_flash(self):
        """Create a red flash effect for violations"""
        # Create a semi-transparent red overlay
        overlay = QWidget(self)
        overlay.setStyleSheet("background-color: rgba(244, 67, 54, 0.3);")
        overlay.resize(self.size())
        overlay.show()
        
        # Animate the overlay
        self.flash_animation = QPropertyAnimation(overlay, b"windowOpacity")
        self.flash_animation.setDuration(500)
        self.flash_animation.setStartValue(0.3)
        self.flash_animation.setEndValue(0.0)
        self.flash_animation.setEasingCurve(QEasingCurve.OutCubic)
        self.flash_animation.finished.connect(overlay.deleteLater)
        self.flash_animation.start()
    
    # Settings persistence
    def save_settings(self):
        """Save window settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("dark_mode", self.dark_mode)
        self.settings.setValue("current_view", self.current_view)
    
    def restore_settings(self):
        """Restore window settings"""
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))
        if self.settings.contains("windowState"):
            self.restoreState(self.settings.value("windowState"))
        if self.settings.contains("dark_mode"):
            self.dark_mode = self.settings.value("dark_mode", True, bool)
        if self.settings.contains("current_view"):
            view_name = self.settings.value("current_view", "live")
            view_index = {"live": 0, "analytics": 1, "violations": 2, "settings": 3}.get(view_name, 0)
            self.tabs.setCurrentIndex(view_index)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings
        self.save_settings()
        
        # Stop video controller
        if hasattr(self.video_controller, 'stop'):
            self.video_controller.stop()
        
        # Accept close event
        event.accept()
