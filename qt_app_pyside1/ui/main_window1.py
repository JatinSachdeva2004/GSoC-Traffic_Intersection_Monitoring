from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QDockWidget, QMessageBox,
    QApplication, QFileDialog, QSplashScreen
)
from PySide6.QtCore import Qt, QTimer, QSettings, QSize, Slot
from PySide6.QtGui import QIcon, QPixmap, QAction
import os
import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

print("✅ Basic PySide6 imports successful")
print("🚀 LOADING MODERN UI - main_window1.py")
print("=" * 50)

# Custom exception handler for Qt
# Ensure Qt is imported before using Qt.qInstallMessageHandler
try:
    from PySide6.QtCore import Qt as QtCoreQt
    if hasattr(QtCoreQt, 'qInstallMessageHandler'):
        def qt_message_handler(mode, context, message):
            print(f"Qt Message: {message} (Mode: {mode})")
        QtCoreQt.qInstallMessageHandler(qt_message_handler)
except Exception as e:
    print(f"⚠️ Could not install Qt message handler: {e}")

# Import UI components with fallback handling
try:
    from ui.fixed_live_tab import LiveTab
    print("✅ Imported LiveTab")
except ImportError as e:
    print(f"⚠️ Could not import LiveTab: {e}")
    # Create a basic fallback LiveTab
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
    from PySide6.QtCore import Signal
    
    class LiveTab(QWidget):
        source_changed = Signal(object)
        video_dropped = Signal(object)
        snapshot_requested = Signal()
        run_requested = Signal(bool)
        
        def __init__(self):
            super().__init__()
            self.current_source = None
            layout = QVBoxLayout(self)
            label = QLabel("Live Tab (Fallback Mode)")
            layout.addWidget(label)
            
        def update_display(self, *args):
            pass
            
        def update_display_np(self, *args):
            pass
            
        def update_stats(self, *args):
            pass

try:
    from ui.analytics_tab import AnalyticsTab
    print("✅ Imported AnalyticsTab")
except ImportError as e:
    print(f"⚠️ Could not import AnalyticsTab: {e}")
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
    
    class AnalyticsTab(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            label = QLabel("Analytics Tab (Fallback Mode)")
            layout.addWidget(label)
            
        def update_analytics(self, *args):
            pass

try:
    from ui.violations_tab import ViolationsTab
    print("✅ Imported ViolationsTab")
except ImportError as e:
    print(f"⚠️ Could not import ViolationsTab: {e}")
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton
    
    class ViolationsTab(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            label = QLabel("Violations Tab (Fallback Mode)")
            self.clear_btn = QPushButton("Clear")
            layout.addWidget(label)
            layout.addWidget(self.clear_btn)
            
        def add_violation(self, *args):
            pass

try:
    from ui.export_tab import ExportTab
    print("✅ Imported ExportTab")
except ImportError as e:
    print(f"⚠️ Could not import ExportTab: {e}")
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox
    
    class ExportTab(QWidget):
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            label = QLabel("Export Tab (Fallback Mode)")
            self.export_format_combo = QComboBox()
            self.export_data_combo = QComboBox()
            self.reset_btn = QPushButton("Reset")
            self.save_config_btn = QPushButton("Save Config")
            self.reload_config_btn = QPushButton("Reload Config")
            self.export_btn = QPushButton("Export")
            
            layout.addWidget(label)
            layout.addWidget(self.export_format_combo)
            layout.addWidget(self.export_data_combo)
            layout.addWidget(self.reset_btn)
            layout.addWidget(self.save_config_btn)
            layout.addWidget(self.reload_config_btn)
            layout.addWidget(self.export_btn)
            
        def update_config_display(self, *args):
            pass
            
        def update_export_preview(self, *args):
            pass
            
        def get_config_from_ui(self):
            return {}

try:
    from ui.config_panel import ConfigPanel
    print("✅ Imported ConfigPanel")
except ImportError as e:
    print(f"⚠️ Could not import ConfigPanel: {e}")
    from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
    from PySide6.QtCore import Signal
    
    class ConfigPanel(QWidget):
        config_changed = Signal(dict)
        theme_toggled = Signal(bool)
        
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout(self)
            label = QLabel("Config Panel (Fallback Mode)")
            layout.addWidget(label)
            
        def set_config(self, *args):
            pass
            
        def reset_config(self):
            pass

# Import controllers with fallback handling
try:
    from controllers.video_controller_new import VideoController
    print("✅ Imported VideoController")
except ImportError as e:
    print(f"⚠️ Could not import VideoController: {e}")
    from PySide6.QtCore import QObject, Signal
    
    class VideoController(QObject):
        frame_ready = Signal(object, object, dict)
        frame_np_ready = Signal(object)
        stats_ready = Signal(dict)
        raw_frame_ready = Signal(object, list, float)
        violation_detected = Signal(dict)
        
        def __init__(self, model_manager=None):
            super().__init__()
            self._running = False
            
        def set_source(self, source):
            print(f"VideoController (fallback): set_source called with {source}")
            return True
            
        def start(self):
            print("VideoController (fallback): start called")
            self._running = True
            
        def stop(self):
            print("VideoController (fallback): stop called")
            self._running = False
            
        def capture_snapshot(self):
            print("VideoController (fallback): capture_snapshot called")
            return None

try:
    from controllers.analytics_controller import AnalyticsController
    print("✅ Imported AnalyticsController")
except ImportError as e:
    print(f"⚠️ Could not import AnalyticsController: {e}")
    from PySide6.QtCore import QObject, Signal
    
    class AnalyticsController(QObject):
        analytics_updated = Signal(dict)
        
        def __init__(self):
            super().__init__()
            
        def process_frame_data(self, *args):
            pass
            
        def clear_statistics(self):
            pass
            
        def register_violation(self, *args):
            pass
            
        def get_analytics(self):
            return {'detection_counts': {}}

try:
    from controllers.performance_overlay import PerformanceOverlay
    print("✅ Imported PerformanceOverlay")
except ImportError as e:
    print(f"⚠️ Could not import PerformanceOverlay: {e}")
    from PySide6.QtWidgets import QWidget
    
    class PerformanceOverlay(QWidget):
        def __init__(self):
            super().__init__()
            self.setVisible(False)
            
        def update_stats(self):
            pass

try:
    from controllers.model_manager import ModelManager
    print("✅ Imported ModelManager")
except ImportError as e:
    print(f"⚠️ Could not import ModelManager: {e}")
    
    class ModelManager:
        def __init__(self, config_file=None):
            print("ModelManager (fallback): initialized")
            
        def update_config(self, config):
            pass

# Import utilities with fallback handling
try:
    from utils.helpers import load_configuration, save_configuration, save_snapshot
    print("✅ Imported utilities")
except ImportError as e:
    print(f"⚠️ Could not import utilities: {e}")
    import json
    import os
    
    def load_configuration(config_file):
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def save_configuration(config, config_file):
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def save_snapshot(frame, file_path):
        try:
            # Try using PySide6's QPixmap to save
            from PySide6.QtGui import QPixmap
            if hasattr(frame, 'shape'):  # NumPy array
                try:
                    import cv2
                    cv2.imwrite(file_path, frame)
                except ImportError:
                    print("OpenCV not available for saving")
                    return None
            else:  # QPixmap or similar
                if hasattr(frame, 'save'):
                    frame.save(file_path)
                else:
                    print("Unknown frame format for saving")
                    return None
            return file_path
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        print("🚀 INITIALIZING MODERN UI - MainWindow1")
        print("=" * 50)
        
        # Initialize settings and configuration
        self.settings = QSettings("OpenVINO", "TrafficMonitoring")
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        self.config = load_configuration(self.config_file)
        
        # Set up UI
        self.setupUI()
        
        # Initialize controllers
        self.setupControllers()
        
        # Connect signals and slots
        self.connectSignals()
        
        # Restore settings
        self.restoreSettings()
        
        # Apply theme - Start with distinctive dark theme
        self.applyTheme(True)  # Start with dark theme
        
        # Show ready message with modern styling
        self.statusBar().showMessage("🚀 Modern UI Ready - All Systems Go!")
        
        print("✅ MODERN UI (MainWindow1) FULLY LOADED!")
        print("=" * 50)
        
    def setupUI(self):
        """Set up the user interface"""
        # Window properties with modern styling
        self.setWindowTitle("🚀 Traffic Monitoring System - MODERN UI (OpenVINO PySide6)")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Add a distinctive window icon or styling
        print("🎨 Setting up MODERN UI interface...")
        
        # Set up central widget with tabs
        self.tabs = QTabWidget()
        
        # Create tabs with enhanced styling
        self.live_tab = LiveTab()
        self.analytics_tab = AnalyticsTab()
        self.violations_tab = ViolationsTab()
        self.export_tab = ExportTab()
        
        # Add tabs to tab widget with modern icons/styling
        self.tabs.addTab(self.live_tab, "🎥 Live Detection")
        self.tabs.addTab(self.analytics_tab, "📊 Analytics")
        self.tabs.addTab(self.violations_tab, "🚨 Violations")
        self.tabs.addTab(self.export_tab, "💾 Export & Config")
        
        # Set central widget
        self.setCentralWidget(self.tabs)
        
        # Create config panel in dock widget with modern styling
        self.config_panel = ConfigPanel()
        dock = QDockWidget("⚙️ Settings Panel", self)
        dock.setObjectName("SettingsDock")  # Set object name to avoid warning
        dock.setWidget(self.config_panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        # Create status bar with modern styling
        self.statusBar().showMessage("🚀 Modern UI Initialized - Ready for Action!")
        
        # Create menu bar
        self.setupMenus()
        
        # Create performance overlay
        self.performance_overlay = PerformanceOverlay()
        
        print("✅ MODERN UI setup completed!")
        
    def setupControllers(self):
        """Set up controllers and models"""
        # Load config from file
        try:
            # Initialize model manager
            self.model_manager = ModelManager(self.config_file)
            print("✅ Model manager initialized")

            # Create video controller
            self.video_controller = VideoController(self.model_manager)
            print("✅ Video controller initialized")

            # Create analytics controller
            self.analytics_controller = AnalyticsController()
            print("✅ Analytics controller initialized")

            # Setup update timer for performance overlay
            if hasattr(self, 'performance_overlay'):
                self.perf_timer = QTimer()
                self.perf_timer.timeout.connect(self.performance_overlay.update_stats)
                self.perf_timer.start(1000)  # Update every second
                print("✅ Performance overlay timer started")
            else:
                print("⚠️ Performance overlay not available")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Error initializing controllers: {str(e)}\n\nPlease check if all required modules are available."
            )
            print(f"❌ Controller initialization error: {e}")
            traceback.print_exc()

    
    def connectSignals(self):
        """Connect signals and slots between components"""
        # Video controller connections - With extra debug
        print("🔌 Connecting video controller signals...")
        try:
            # Connect for UI frame updates (QPixmap-based)
            if hasattr(self.live_tab, 'update_display'):
                self.video_controller.frame_ready.connect(self.live_tab.update_display, Qt.QueuedConnection)
                print("✅ Connected frame_ready signal")
            else:
                print("⚠️ live_tab.update_display method not found")
                
            # Connect for direct NumPy frame display (critical for live video)
            if hasattr(self.live_tab, 'update_display_np'):
                self.video_controller.frame_np_ready.connect(self.live_tab.update_display_np, Qt.QueuedConnection)
                print("✅ Connected frame_np_ready signal")
            else:
                print("⚠️ live_tab.update_display_np method not found")
                
            # Connect stats signal
            if hasattr(self.live_tab, 'update_stats'):
                self.video_controller.stats_ready.connect(self.live_tab.update_stats, Qt.QueuedConnection)
                print("✅ Connected stats_ready to live_tab")
            else:
                print("⚠️ live_tab.update_stats method not found")
                
            # Also connect stats signal to update traffic light status in main window
            self.video_controller.stats_ready.connect(self.update_traffic_light_status, Qt.QueuedConnection)
            print("✅ Connected stats_ready signals")
              
            # Connect raw frame data for analytics
            if hasattr(self.analytics_controller, 'process_frame_data'):
                self.video_controller.raw_frame_ready.connect(self.analytics_controller.process_frame_data)
                print("✅ Connected raw_frame_ready signal")
            else:
                print("⚠️ analytics_controller.process_frame_data method not found")
            
            # Connect violation detection signal
            try:
                self.video_controller.violation_detected.connect(self.handle_violation_detected, Qt.QueuedConnection)
                print("✅ Connected violation_detected signal")
            except Exception as e:
                print(f"⚠️ Could not connect violation signal: {e}")
        except Exception as e:
            print(f"❌ Error connecting video controller signals: {e}")
            traceback.print_exc()
        
        # Live tab connections - with safety checks
        try:
            if hasattr(self.live_tab, 'source_changed'):
                self.live_tab.source_changed.connect(self.video_controller.set_source)
                print("✅ Connected live_tab.source_changed")
            if hasattr(self.live_tab, 'video_dropped'):
                self.live_tab.video_dropped.connect(self.video_controller.set_source)
                print("✅ Connected live_tab.video_dropped")
            if hasattr(self.live_tab, 'snapshot_requested'):
                self.live_tab.snapshot_requested.connect(self.take_snapshot)
                print("✅ Connected live_tab.snapshot_requested")
            if hasattr(self.live_tab, 'run_requested'):
                self.live_tab.run_requested.connect(self.toggle_video_processing)
                print("✅ Connected live_tab.run_requested")
        except Exception as e:
            print(f"⚠️ Error connecting live_tab signals: {e}")
        
        # Config panel connections - with safety checks
        try:
            if hasattr(self.config_panel, 'config_changed'):
                self.config_panel.config_changed.connect(self.apply_config)
                print("✅ Connected config_panel.config_changed")
            if hasattr(self.config_panel, 'theme_toggled'):
                self.config_panel.theme_toggled.connect(self.applyTheme)
                print("✅ Connected config_panel.theme_toggled")
        except Exception as e:
            print(f"⚠️ Error connecting config_panel signals: {e}")
        
        # Analytics controller connections - with safety checks
        try:
            if hasattr(self.analytics_controller, 'analytics_updated'):
                if hasattr(self.analytics_tab, 'update_analytics'):
                    self.analytics_controller.analytics_updated.connect(self.analytics_tab.update_analytics)
                    print("✅ Connected analytics_controller to analytics_tab")
                if hasattr(self.export_tab, 'update_export_preview'):
                    self.analytics_controller.analytics_updated.connect(self.export_tab.update_export_preview)
                    print("✅ Connected analytics_controller to export_tab")
        except Exception as e:
            print(f"⚠️ Error connecting analytics_controller signals: {e}")
        
        # Tab-specific connections - with safety checks
        try:
            if hasattr(self.violations_tab, 'clear_btn') and hasattr(self.analytics_controller, 'clear_statistics'):
                self.violations_tab.clear_btn.clicked.connect(self.analytics_controller.clear_statistics)
                print("✅ Connected violations_tab.clear_btn")
                
            if hasattr(self.export_tab, 'reset_btn') and hasattr(self.config_panel, 'reset_config'):
                self.export_tab.reset_btn.clicked.connect(self.config_panel.reset_config)
                print("✅ Connected export_tab.reset_btn")
                
            if hasattr(self.export_tab, 'save_config_btn'):
                self.export_tab.save_config_btn.clicked.connect(self.save_config)
                print("✅ Connected export_tab.save_config_btn")
                
            if hasattr(self.export_tab, 'reload_config_btn'):
                self.export_tab.reload_config_btn.clicked.connect(self.load_config)
                print("✅ Connected export_tab.reload_config_btn")
                
            if hasattr(self.export_tab, 'export_btn'):
                self.export_tab.export_btn.clicked.connect(self.export_data)
                print("✅ Connected export_tab.export_btn")
        except Exception as e:
            print(f"⚠️ Error connecting tab-specific signals: {e}")
            
        print("🔌 Signal connection process completed")
        
    def setupMenus(self):
        """Set up application menus"""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        open_action = QAction("&Open Video...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        snapshot_action = QAction("Take &Snapshot", self)
        snapshot_action.setShortcut("Ctrl+S")
        snapshot_action.triggered.connect(self.take_snapshot)
        file_menu.addAction(snapshot_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = self.menuBar().addMenu("&View")
        
        toggle_config_action = QAction("Show/Hide &Settings Panel", self)
        toggle_config_action.setShortcut("F4")
        toggle_config_action.triggered.connect(self.toggle_config_panel)
        view_menu.addAction(toggle_config_action)
        
        toggle_perf_action = QAction("Show/Hide &Performance Overlay", self)
        toggle_perf_action.setShortcut("F5")
        toggle_perf_action.triggered.connect(self.toggle_performance_overlay)
        view_menu.addAction(toggle_perf_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
    @Slot(dict)
    def apply_config(self, config):
        """
        Apply configuration changes.
        
        Args:
            config: Configuration dictionary
        """
        # Update configuration
        if not config:
            return
            
        # Update config
        for section in config:
            if section in self.config:
                self.config[section].update(config[section])
            else:
                self.config[section] = config[section]
        
        # Update model manager
        if self.model_manager:
            self.model_manager.update_config(self.config)
        
        # Save config to file
        save_configuration(self.config, self.config_file)
        
        # Update export tab
        self.export_tab.update_config_display(self.config)
        
        # Update status
        self.statusBar().showMessage("Configuration applied", 2000)
    
    @Slot()
    def load_config(self):
        """Load configuration from file"""
        # Ask for confirmation if needed
        if self.video_controller and self.video_controller._running:
            reply = QMessageBox.question(
                self,
                "Reload Configuration",
                "Reloading configuration will stop current processing. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
                
            # Stop processing
            self.video_controller.stop()
        
        # Load config
        self.config = load_configuration(self.config_file)
        
        # Update UI
        self.config_panel.set_config(self.config)
        self.export_tab.update_config_display(self.config)
        
        # Update model manager
        if self.model_manager:
            self.model_manager.update_config(self.config)
        
        # Update status
        self.statusBar().showMessage("Configuration loaded", 2000)
    
    @Slot()
    def save_config(self):
        """Save configuration to file"""
        # Get config from UI
        ui_config = self.export_tab.get_config_from_ui()
        for section in ui_config:
            if section in self.config:
                self.config[section].update(ui_config[section])
            else:
                self.config[section] = ui_config[section]
        
        # Save to file
        if save_configuration(self.config, self.config_file):
            self.statusBar().showMessage("Configuration saved", 2000)
        else:
            self.statusBar().showMessage("Error saving configuration", 2000)
            
        # Update model manager
        if self.model_manager:
            self.model_manager.update_config(self.config)
    
    @Slot()
    def open_video_file(self):
        """Open video file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if file_path:
            # Update live tab
            self.live_tab.source_changed.emit(file_path)
            
            # Update status
            self.statusBar().showMessage(f"Loaded video: {os.path.basename(file_path)}")
    
    @Slot()
    def take_snapshot(self):
        """Take snapshot of current frame"""
        if self.video_controller:
            # Get current frame
            frame = self.video_controller.capture_snapshot()
            
            if frame is not None:
                # Save frame to file
                save_dir = self.settings.value("snapshot_dir", ".")
                file_path = os.path.join(save_dir, "snapshot_" + 
                                        str(int(time.time())) + ".jpg")
                
                saved_path = save_snapshot(frame, file_path)
                
                if saved_path:
                    self.statusBar().showMessage(f"Snapshot saved: {saved_path}", 3000)
                else:
                    self.statusBar().showMessage("Error saving snapshot", 3000)
            else:
                self.statusBar().showMessage("No frame to capture", 3000)
    
    @Slot()
    def toggle_config_panel(self):
        """Toggle configuration panel visibility"""
        dock_widgets = self.findChildren(QDockWidget)
        for dock in dock_widgets:
            dock.setVisible(not dock.isVisible())
    
    @Slot()
    def toggle_performance_overlay(self):
        """Toggle performance overlay visibility"""
        if self.performance_overlay.isVisible():
            self.performance_overlay.hide()
        else:
            # Position in the corner
            self.performance_overlay.move(self.pos().x() + 10, self.pos().y() + 30)
            self.performance_overlay.show()
    
    @Slot(bool)
    def applyTheme(self, dark_theme):
        """
        Apply light or dark theme.
        
        Args:
            dark_theme: True for dark theme, False for light theme
        """
        if dark_theme:
            # Apply a modern dark theme with distinctive styling
            dark_stylesheet = """
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
                border: none;
            }
            
            QTabWidget::pane {
                border: 1px solid #404040;
                background-color: #2d2d2d;
                border-radius: 8px;
            }
            
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                font-size: 11px;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            
            QTabBar::tab:hover {
                background-color: #555555;
            }
            
            QStatusBar {
                background-color: #333333;
                color: #ffffff;
                border-top: 1px solid #555555;
                font-weight: bold;
            }
            
            QMenuBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border-bottom: 1px solid #555555;
                padding: 4px;
            }
            
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 16px;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background-color: #0078d4;
            }
            
            QMenu {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            
            QMenu::item {
                padding: 8px 24px;
            }
            
            QMenu::item:selected {
                background-color: #0078d4;
            }
            
            QDockWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                titlebar-close-icon: none;
                titlebar-normal-icon: none;
                border: 1px solid #555555;
                border-radius: 4px;
            }
            
            QDockWidget::title {
                background-color: #404040;
                color: #ffffff;
                padding: 8px;
                text-align: center;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QWidget {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            
            QLabel {
                color: #ffffff;
                font-size: 11px;
            }
            
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10px;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            
            QPushButton:pressed {
                background-color: #005a9e;
            }
            
            QComboBox {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
                font-size: 10px;
            }
            
            QComboBox::drop-down {
                border: none;
                background-color: #0078d4;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #ffffff;
                border-top: none;
                border-left: none;
                width: 6px;
                height: 6px;
                margin-right: 4px;
                transform: rotate(45deg);
            }
            
            QComboBox QAbstractItemView {
                background-color: #404040;
                color: #ffffff;
                selection-background-color: #0078d4;
                border: 1px solid #555555;
            }
            
            /* Make the title bar more distinctive */
            QMainWindow::title {
                background-color: #1e1e1e;
                color: #0078d4;
                font-weight: bold;
                font-size: 14px;
            }
            """
            self.setStyleSheet(dark_stylesheet)
            
            # Also update window title to show it's the modern UI
            self.setWindowTitle("🚀 Traffic Monitoring System - MODERN UI (OpenVINO PySide6)")
            
        else:
            # Light theme with modern styling
            light_stylesheet = """
            QMainWindow {
                background-color: #f5f5f5;
                color: #333333;
            }
            
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
                border-radius: 8px;
            }
            
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
            
            QStatusBar {
                background-color: #e0e0e0;
                color: #333333;
                border-top: 1px solid #cccccc;
                font-weight: bold;
            }
            
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #106ebe;
            }
            """
            self.setStyleSheet(light_stylesheet)
            self.setWindowTitle("☀️ Traffic Monitoring System - LIGHT UI (OpenVINO PySide6)")
            
        # Update status bar to show theme change
        theme_name = "🌙 DARK MODERN" if dark_theme else "☀️ LIGHT MODERN"
        self.statusBar().showMessage(f"Theme applied: {theme_name} UI", 3000)
    
    @Slot()
    def export_data(self):
        """Export data to file"""
        export_format = self.export_tab.export_format_combo.currentText()
        export_data = self.export_tab.export_data_combo.currentText()
        
        # Get file type filter based on format
        if export_format == "CSV":
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
        elif export_format == "JSON":
            file_filter = "JSON Files (*.json)"
            default_ext = ".json"
        elif export_format == "Excel":
            file_filter = "Excel Files (*.xlsx)"
            default_ext = ".xlsx"
        elif export_format == "PDF Report":
            file_filter = "PDF Files (*.pdf)"
            default_ext = ".pdf"
        else:
            file_filter = "All Files (*)"
            default_ext = ".txt"
        
        # Get save path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            f"traffic_data{default_ext}",
            file_filter
        )
        
        if not file_path:
            return
            
        try:
            # Get analytics data
            analytics = self.analytics_controller.get_analytics()
            
            # Export based on format
            if export_format == "CSV":
                try:
                    from utils.helpers import create_export_csv
                    result = create_export_csv(analytics['detection_counts'], file_path)
                except ImportError:
                    print("CSV export not available - utils.helpers not found")
                    result = False
            elif export_format == "JSON":
                try:
                    from utils.helpers import create_export_json
                    result = create_export_json(analytics, file_path)
                except ImportError:
                    # Fallback JSON export
                    try:
                        with open(file_path, 'w') as f:
                            json.dump(analytics, f, indent=2, default=str)
                        result = True
                    except Exception as e:
                        print(f"JSON export error: {e}")
                        result = False
            elif export_format == "Excel":
                # Requires openpyxl
                try:
                    import pandas as pd
                    df = pd.DataFrame({
                        'Class': list(analytics['detection_counts'].keys()),
                        'Count': list(analytics['detection_counts'].values())
                    })
                    df.to_excel(file_path, index=False)
                    result = True
                except Exception as e:
                    print(f"Excel export error: {e}")
                    result = False
            else:
                # Not implemented
                QMessageBox.information(
                    self,
                    "Not Implemented",
                    f"Export to {export_format} is not yet implemented."
                )
                return
                
            if result:
                self.statusBar().showMessage(f"Data exported to {file_path}", 3000)
            else:
                self.statusBar().showMessage("Error exporting data", 3000)
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Error exporting data: {str(e)}"
            )
    
    @Slot()
    def show_about_dialog(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Traffic Monitoring System",
            "<h3>Traffic Monitoring System</h3>"
            "<p>Based on OpenVINO™ and PySide6</p>"
            "<p>Version 1.0.0</p>"
            "<p>© 2025 GSOC Project</p>"
        )
    @Slot(bool)
    def toggle_video_processing(self, start):
        """
        Start or stop video processing.
        
        Args:
            start: True to start processing, False to stop
        """
        if self.video_controller:
            if start:
                try:
                    # Make sure the source is correctly set to what the LiveTab has
                    current_source = self.live_tab.current_source
                    print(f"DEBUG: MainWindow toggle_processing with source: {current_source} (type: {type(current_source)})")
                    
                    # Validate source
                    if current_source is None:
                        self.statusBar().showMessage("Error: No valid source selected")
                        return
                        
                    # For file sources, verify file exists
                    if isinstance(current_source, str) and not current_source.isdigit():
                        if not os.path.exists(current_source):
                            self.statusBar().showMessage(f"Error: File not found: {current_source}")
                            return
                    
                    # Ensure the source is set before starting
                    print(f"🎥 Setting video controller source to: {current_source}")
                    self.video_controller.set_source(current_source)
                    
                    # Now start processing after a short delay to ensure source is set
                    print("⏱️ Scheduling video processing start after 200ms delay...")
                    QTimer.singleShot(200, lambda: self._start_video_processing())
                    
                    source_desc = f"file: {os.path.basename(current_source)}" if isinstance(current_source, str) and os.path.exists(current_source) else f"camera: {current_source}"
                    self.statusBar().showMessage(f"Video processing started with {source_desc}")
                except Exception as e:
                    print(f"❌ Error starting video: {e}")
                    traceback.print_exc()
                    self.statusBar().showMessage(f"Error: {str(e)}")
            else:
                try:
                    print("🛑 Stopping video processing...")
                    self.video_controller.stop()
                    print("✅ Video controller stopped")
                    self.statusBar().showMessage("Video processing stopped")
                except Exception as e:
                    print(f"❌ Error stopping video: {e}")
                    traceback.print_exc()
                    
    def _start_video_processing(self):
        """Actual video processing start with extra error handling"""
        try:
            print("🚀 Starting video controller...")
            self.video_controller.start()
            print("✅ Video controller started successfully")
        except Exception as e:
            print(f"❌ Error in video processing start: {e}")
            traceback.print_exc()
            self.statusBar().showMessage(f"Video processing error: {str(e)}")
                
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop processing
        if self.video_controller and self.video_controller._running:
            self.video_controller.stop()
            
        # Save settings
        self.saveSettings()
        
        # Accept close event
        event.accept()
    
    def restoreSettings(self):
        """Restore application settings"""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def saveSettings(self):
        """Save application settings"""
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())
        
        # Save window state
        self.settings.setValue("windowState", self.saveState())
        
        # Save current directory as snapshot directory
        self.settings.setValue("snapshot_dir", os.getcwd())
    @Slot(dict)
    def update_traffic_light_status(self, stats):
        """Update status bar with traffic light information if detected"""
        traffic_light_info = stats.get('traffic_light_color', 'unknown')
        
        # Handle both string and dictionary return formats
        if isinstance(traffic_light_info, dict):
            traffic_light_color = traffic_light_info.get('color', 'unknown')
            confidence = traffic_light_info.get('confidence', 0.0)
            confidence_str = f" (Confidence: {confidence:.2f})" if confidence > 0 else ""
        else:
            traffic_light_color = traffic_light_info
            confidence_str = ""
            
        if traffic_light_color != 'unknown':
            current_message = self.statusBar().currentMessage()
            if not current_message or "Traffic Light" not in current_message:
                # Handle both dictionary and string formats
                if isinstance(traffic_light_color, dict):
                    color_text = traffic_light_color.get("color", "unknown").upper()
                else:
                    color_text = str(traffic_light_color).upper()
                self.statusBar().showMessage(f"Traffic Light: {color_text}{confidence_str}")
    @Slot(dict)
    def handle_violation_detected(self, violation):
        """Handle a detected traffic violation"""
        try:
            # Get track ID safely
            track_id = violation.get('track_id', violation.get('id', 'Unknown'))
            timestamp = violation.get('timestamp', datetime.now())
            
            # Flash red status message
            self.statusBar().showMessage(f"🚨 RED LIGHT VIOLATION DETECTED - Vehicle ID: {track_id}", 5000)
            
            # Add to violations tab
            if hasattr(self.violations_tab, 'add_violation'):
                self.violations_tab.add_violation(violation)
            else:
                print("⚠️ violations_tab.add_violation method not found")
            
            # Update analytics
            if self.analytics_controller and hasattr(self.analytics_controller, 'register_violation'):
                self.analytics_controller.register_violation(violation)
            else:
                print("⚠️ analytics_controller.register_violation method not found")
                
            print(f"🚨 Violation processed: Track ID={track_id} at {timestamp}")
        except Exception as e:
            print(f"❌ Error handling violation: {e}")
            traceback.print_exc()
