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
from pathlib import Path

# Custom exception handler for Qt
def qt_message_handler(mode, context, message):
    print(f"Qt Message: {message} (Mode: {mode})")

# Install custom handler for Qt messages
if hasattr(Qt, 'qInstallMessageHandler'):
    Qt.qInstallMessageHandler(qt_message_handler)

# Import UI components
from ..ui.fixed_live_tab import LiveTab  # Using fixed version
from ..ui.analytics_tab import AnalyticsTab
from ..ui.violations_tab import ViolationsTab
from ..ui.export_tab import ExportTab
from ..ui.config_panel import ConfigPanel

# Import controllers
from ..controllers.video_controller_new import VideoController
from ..controllers.analytics_controller import AnalyticsController
from ..controllers.performance_overlay import PerformanceOverlay
from ..controllers.model_manager import ModelManager

# Import utilities
from ..utils.helpers import load_configuration, save_configuration, save_snapshot

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
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
        
        # Apply theme
        self.applyTheme(True)  # Start with dark theme
        
        # Show ready message
        self.statusBar().showMessage("Ready")
        
    def setupUI(self):
        """Set up the user interface"""
        # Window properties
        self.setWindowTitle("Traffic Monitoring System (OpenVINO PySide6)")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Set up central widget with tabs
        self.tabs = QTabWidget()
        
        # Create tabs
        self.live_tab = LiveTab()
        self.analytics_tab = AnalyticsTab()
        self.violations_tab = ViolationsTab()
        self.export_tab = ExportTab()
        
        # Add tabs to tab widget
        self.tabs.addTab(self.live_tab, "Live Detection")
        self.tabs.addTab(self.analytics_tab, "Analytics")
        self.tabs.addTab(self.violations_tab, "Violations")
        self.tabs.addTab(self.export_tab, "Export & Config")
        
        # Set central widget
        self.setCentralWidget(self.tabs)
          # Create config panel in dock widget
        self.config_panel = ConfigPanel()
        dock = QDockWidget("Settings", self)
        dock.setObjectName("SettingsDock")  # Set object name to avoid warning
        dock.setWidget(self.config_panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
        # Create status bar
        self.statusBar().showMessage("Initializing...")
        
        # Create menu bar
        self.setupMenus()
        
        # Create performance overlay
        self.performance_overlay = PerformanceOverlay()
        
    def setupControllers(self):
        """Set up controllers and models"""
        # Load config from file
        try:
            # Initialize model manager
            self.model_manager = ModelManager(self.config_file)

            # Create video controller
            self.video_controller = VideoController(self.model_manager)

            # Create analytics controller
            self.analytics_controller = AnalyticsController()

            # Setup update timer for performance overlay
            self.perf_timer = QTimer()
            self.perf_timer.timeout.connect(self.performance_overlay.update_stats)
            self.perf_timer.start(1000)  # Update every second

        except Exception as e:
            QMessageBox.critical(
                self,
                "Initialization Error",
                f"Error initializing controllers: {str(e)}"
            )
            print(f"Error details: {e}")

    
    def connectSignals(self):
        """Connect signals and slots between components"""        # Video controller connections - With extra debug
        print("🔌 Connecting video controller signals...")
        try:
            # Connect for UI frame updates (QPixmap-based)
            self.video_controller.frame_ready.connect(self.live_tab.update_display, Qt.QueuedConnection)
            print("✅ Connected frame_ready signal")            # Connect for direct NumPy frame display (critical for live video)
            try:
                self.video_controller.frame_np_ready.connect(self.live_tab.update_display_np, Qt.QueuedConnection)
                print("✅ Connected frame_np_ready signal")
                # PySide6 doesn't have isConnected method, so let's just confirm the connection works
                print("🔌 frame_np_ready connection should be established")
            except Exception as e:
                print(f"❌ Error connecting frame_np_ready signal: {e}")
                import traceback
                traceback.print_exc()
              # Connect stats signal
            self.video_controller.stats_ready.connect(self.live_tab.update_stats, Qt.QueuedConnection)
            # Also connect stats signal to update traffic light status in main window
            self.video_controller.stats_ready.connect(self.update_traffic_light_status, Qt.QueuedConnection)
            print("✅ Connected stats_ready signals")
              # Connect raw frame data for analytics
            self.video_controller.raw_frame_ready.connect(self.analytics_controller.process_frame_data)
            print("✅ Connected raw_frame_ready signal")
            
            # Connect for traffic light status updates
            self.video_controller.stats_ready.connect(self.update_traffic_light_status, Qt.QueuedConnection)
            print("✅ Connected stats_ready signal to update_traffic_light_status")
            
            # Connect violation detection signal
            try:
                self.video_controller.violation_detected.connect(self.handle_violation_detected, Qt.QueuedConnection)
                print("✅ Connected violation_detected signal")
            except Exception as e:
                print(f"⚠️ Could not connect violation signal: {e}")
        except Exception as e:
            print(f"❌ Error connecting signals: {e}")
            import traceback
            traceback.print_exc()
        
        # Live tab connections
        self.live_tab.source_changed.connect(self.video_controller.set_source)
        self.live_tab.video_dropped.connect(self.video_controller.set_source)
        self.live_tab.snapshot_requested.connect(self.take_snapshot)
        self.live_tab.run_requested.connect(self.toggle_video_processing)
        
        # Config panel connections
        self.config_panel.config_changed.connect(self.apply_config)
        self.config_panel.theme_toggled.connect(self.applyTheme)
        
        # Analytics controller connections
        self.analytics_controller.analytics_updated.connect(self.analytics_tab.update_analytics)
        self.analytics_controller.analytics_updated.connect(self.export_tab.update_export_preview)
        
        # Tab-specific connections
        self.violations_tab.clear_btn.clicked.connect(self.analytics_controller.clear_statistics)
        self.export_tab.reset_btn.clicked.connect(self.config_panel.reset_config)
        self.export_tab.save_config_btn.clicked.connect(self.save_config)
        self.export_tab.reload_config_btn.clicked.connect(self.load_config)
        self.export_tab.export_btn.clicked.connect(self.export_data)
        
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
        
        # Update config
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
            # Load dark theme stylesheet
            theme_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources", "themes", "dark.qss"
            )
        else:
            # Load light theme stylesheet
            theme_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "resources", "themes", "light.qss"
            )
            
        # Apply theme if file exists
        if os.path.exists(theme_file):
            with open(theme_file, "r") as f:
                self.setStyleSheet(f.read())
        else:
            # Fallback to built-in style
            self.setStyleSheet("")
    
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
                from ..utils.helpers import create_export_csv
                result = create_export_csv(analytics['detection_counts'], file_path)
            elif export_format == "JSON":
                from ..utils.helpers import create_export_json
                result = create_export_json(analytics, file_path)
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
            # Flash red status message
            self.statusBar().showMessage(f"🚨 RED LIGHT VIOLATION DETECTED - Vehicle ID: {violation['track_id']}", 5000)
            
            # Add to violations tab
            self.violations_tab.add_violation(violation)
            
            # Update analytics
            if self.analytics_controller:
                self.analytics_controller.register_violation(violation)
                
            print(f"🚨 Violation processed: {violation['id']} at {violation['timestamp']}")
        except Exception as e:
            print(f"❌ Error handling violation: {e}")
            import traceback
            traceback.print_exc()
