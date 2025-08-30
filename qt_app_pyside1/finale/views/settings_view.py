"""
Settings View - Application configuration and preferences
Manages all application settings, model configurations, and system preferences.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFrame, QScrollArea, QTabWidget,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QSlider, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
    QFormLayout, QButtonGroup, QRadioButton
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QSettings, QThread
from PySide6.QtGui import QFont, QPixmap

import os
import json
import sys
from pathlib import Path

# Import finale components
from ..styles import FinaleStyles, MaterialColors
from ..icons import FinaleIcons
from qt_app_pyside.ui.config_panel import ConfigPanel
from qt_app_pyside.utils.helpers import load_configuration, save_configuration
from qt_app_pyside.utils.helpers import format_timestamp, format_duration

class ModelConfigWidget(QGroupBox):
    """
    Widget for configuring AI models and detection parameters.
    """
    
    config_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__("AI Model Configuration", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup model configuration UI"""
        layout = QFormLayout(self)
        
        # Vehicle detection model
        self.vehicle_model_edit = QLineEdit()
        self.vehicle_model_edit.setPlaceholderText("Path to vehicle detection model...")
        
        vehicle_browse_btn = QPushButton(FinaleIcons.get_icon("folder"), "")
        vehicle_browse_btn.setFixedSize(32, 32)
        vehicle_browse_btn.clicked.connect(lambda: self.browse_model("vehicle"))
        
        vehicle_layout = QHBoxLayout()
        vehicle_layout.addWidget(self.vehicle_model_edit)
        vehicle_layout.addWidget(vehicle_browse_btn)
        
        layout.addRow("Vehicle Model:", vehicle_layout)
        
        # Traffic light detection model
        self.traffic_model_edit = QLineEdit()
        self.traffic_model_edit.setPlaceholderText("Path to traffic light model...")
        
        traffic_browse_btn = QPushButton(FinaleIcons.get_icon("folder"), "")
        traffic_browse_btn.setFixedSize(32, 32)
        traffic_browse_btn.clicked.connect(lambda: self.browse_model("traffic"))
        
        traffic_layout = QHBoxLayout()
        traffic_layout.addWidget(self.traffic_model_edit)
        traffic_layout.addWidget(traffic_browse_btn)
        
        layout.addRow("Traffic Light Model:", traffic_layout)
        
        # Detection parameters
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.3)
        self.confidence_spin.setSuffix(" (30%)")
        layout.addRow("Confidence Threshold:", self.confidence_spin)
        
        self.nms_spin = QDoubleSpinBox()
        self.nms_spin.setRange(0.1, 1.0)
        self.nms_spin.setSingleStep(0.05)
        self.nms_spin.setValue(0.45)
        layout.addRow("NMS Threshold:", self.nms_spin)
        
        self.max_detections_spin = QSpinBox()
        self.max_detections_spin.setRange(10, 1000)
        self.max_detections_spin.setValue(100)
        layout.addRow("Max Detections:", self.max_detections_spin)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU", "AUTO"])
        layout.addRow("Device:", self.device_combo)
        
        # Model optimization
        self.optimize_check = QCheckBox("Enable Model Optimization")
        self.optimize_check.setChecked(True)
        layout.addRow(self.optimize_check)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())
    
    @Slot()
    def browse_model(self, model_type):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select {model_type.title()} Model", "",
            "Model Files (*.xml *.onnx *.pt *.bin);;All Files (*)"
        )
        
        if file_path:
            if model_type == "vehicle":
                self.vehicle_model_edit.setText(file_path)
            elif model_type == "traffic":
                self.traffic_model_edit.setText(file_path)
    
    def get_config(self):
        """Get current model configuration"""
        return {
            'vehicle_model': self.vehicle_model_edit.text(),
            'traffic_model': self.traffic_model_edit.text(),
            'confidence_threshold': self.confidence_spin.value(),
            'nms_threshold': self.nms_spin.value(),
            'max_detections': self.max_detections_spin.value(),
            'device': self.device_combo.currentText(),
            'optimize_model': self.optimize_check.isChecked()
        }
    
    def set_config(self, config):
        """Set model configuration"""
        self.vehicle_model_edit.setText(config.get('vehicle_model', ''))
        self.traffic_model_edit.setText(config.get('traffic_model', ''))
        self.confidence_spin.setValue(config.get('confidence_threshold', 0.3))
        self.nms_spin.setValue(config.get('nms_threshold', 0.45))
        self.max_detections_spin.setValue(config.get('max_detections', 100))
        self.device_combo.setCurrentText(config.get('device', 'CPU'))
        self.optimize_check.setChecked(config.get('optimize_model', True))

class ViolationConfigWidget(QGroupBox):
    """
    Widget for configuring violation detection parameters.
    """
    
    def __init__(self, parent=None):
        super().__init__("Violation Detection", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup violation configuration UI"""
        layout = QFormLayout(self)
        
        # Red light violation
        self.red_light_check = QCheckBox("Enable Red Light Detection")
        self.red_light_check.setChecked(True)
        layout.addRow(self.red_light_check)
        
        self.red_light_sensitivity = QSlider(Qt.Horizontal)
        self.red_light_sensitivity.setRange(1, 10)
        self.red_light_sensitivity.setValue(5)
        layout.addRow("Red Light Sensitivity:", self.red_light_sensitivity)
        
        # Speed violation
        self.speed_check = QCheckBox("Enable Speed Detection")
        self.speed_check.setChecked(True)
        layout.addRow(self.speed_check)
        
        self.speed_limit_spin = QSpinBox()
        self.speed_limit_spin.setRange(10, 200)
        self.speed_limit_spin.setValue(50)
        self.speed_limit_spin.setSuffix(" km/h")
        layout.addRow("Speed Limit:", self.speed_limit_spin)
        
        self.speed_tolerance_spin = QSpinBox()
        self.speed_tolerance_spin.setRange(0, 20)
        self.speed_tolerance_spin.setValue(5)
        self.speed_tolerance_spin.setSuffix(" km/h")
        layout.addRow("Speed Tolerance:", self.speed_tolerance_spin)
        
        # Wrong lane detection
        self.wrong_lane_check = QCheckBox("Enable Wrong Lane Detection")
        self.wrong_lane_check.setChecked(True)
        layout.addRow(self.wrong_lane_check)
        
        # Helmet detection
        self.helmet_check = QCheckBox("Enable Helmet Detection")
        self.helmet_check.setChecked(False)
        layout.addRow(self.helmet_check)
        
        # Violation zone setup
        self.zone_setup_btn = QPushButton(FinaleIcons.get_icon("map"), "Setup Violation Zones")
        layout.addRow(self.zone_setup_btn)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())

class UIPreferencesWidget(QGroupBox):
    """
    Widget for UI preferences and appearance settings.
    """
    
    theme_changed = Signal(bool)  # dark_mode
    
    def __init__(self, parent=None):
        super().__init__("User Interface", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI preferences"""
        layout = QFormLayout(self)
        
        # Theme selection
        theme_group = QButtonGroup(self)
        self.dark_radio = QRadioButton("Dark Theme")
        self.light_radio = QRadioButton("Light Theme")
        self.auto_radio = QRadioButton("Auto (System)")
        
        self.dark_radio.setChecked(True)  # Default to dark
        
        theme_group.addButton(self.dark_radio)
        theme_group.addButton(self.light_radio)
        theme_group.addButton(self.auto_radio)
        
        theme_layout = QVBoxLayout()
        theme_layout.addWidget(self.dark_radio)
        theme_layout.addWidget(self.light_radio)
        theme_layout.addWidget(self.auto_radio)
        
        layout.addRow("Theme:", theme_layout)
        
        # Language selection
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Español", "Français", "Deutsch", "العربية"])
        layout.addRow("Language:", self.language_combo)
        
        # Font size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 16)
        self.font_size_spin.setValue(9)
        layout.addRow("Font Size:", self.font_size_spin)
        
        # Animations
        self.animations_check = QCheckBox("Enable Animations")
        self.animations_check.setChecked(True)
        layout.addRow(self.animations_check)
        
        # Sound notifications
        self.sound_check = QCheckBox("Sound Notifications")
        self.sound_check.setChecked(True)
        layout.addRow(self.sound_check)
        
        # Auto-save
        self.autosave_check = QCheckBox("Auto-save Configuration")
        self.autosave_check.setChecked(True)
        layout.addRow(self.autosave_check)
        
        # Update interval
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(100, 5000)
        self.update_interval_spin.setValue(1000)
        self.update_interval_spin.setSuffix(" ms")
        layout.addRow("Update Interval:", self.update_interval_spin)
        
        # Connect theme signals
        self.dark_radio.toggled.connect(lambda checked: self.theme_changed.emit(True) if checked else None)
        self.light_radio.toggled.connect(lambda checked: self.theme_changed.emit(False) if checked else None)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())

class PerformanceWidget(QGroupBox):
    """
    Widget for performance and system settings.
    """
    
    def __init__(self, parent=None):
        super().__init__("Performance", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup performance settings"""
        layout = QFormLayout(self)
        
        # Processing threads
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 16)
        self.threads_spin.setValue(4)
        layout.addRow("Processing Threads:", self.threads_spin)
        
        # Frame buffer size
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(1, 100)
        self.buffer_size_spin.setValue(10)
        layout.addRow("Frame Buffer Size:", self.buffer_size_spin)
        
        # Memory limit
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(512, 8192)
        self.memory_limit_spin.setValue(2048)
        self.memory_limit_spin.setSuffix(" MB")
        layout.addRow("Memory Limit:", self.memory_limit_spin)
        
        # GPU acceleration
        self.gpu_check = QCheckBox("Enable GPU Acceleration")
        self.gpu_check.setChecked(False)
        layout.addRow(self.gpu_check)
        
        # Performance mode
        self.performance_combo = QComboBox()
        self.performance_combo.addItems(["Balanced", "Performance", "Power Save"])
        layout.addRow("Performance Mode:", self.performance_combo)
        
        # Logging level
        self.logging_combo = QComboBox()
        self.logging_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.logging_combo.setCurrentText("INFO")
        layout.addRow("Logging Level:", self.logging_combo)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())

class DataManagementWidget(QGroupBox):
    """
    Widget for data storage and export settings.
    """
    
    def __init__(self, parent=None):
        super().__init__("Data Management", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup data management settings"""
        layout = QFormLayout(self)
        
        # Data directory
        self.data_dir_edit = QLineEdit()
        self.data_dir_edit.setPlaceholderText("Data storage directory...")
        
        data_browse_btn = QPushButton(FinaleIcons.get_icon("folder"), "")
        data_browse_btn.setFixedSize(32, 32)
        data_browse_btn.clicked.connect(self.browse_data_directory)
        
        data_layout = QHBoxLayout()
        data_layout.addWidget(self.data_dir_edit)
        data_layout.addWidget(data_browse_btn)
        
        layout.addRow("Data Directory:", data_layout)
        
        # Auto-export
        self.auto_export_check = QCheckBox("Auto-export Violations")
        layout.addRow(self.auto_export_check)
        
        # Export format
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["JSON", "CSV", "XML", "PDF"])
        layout.addRow("Export Format:", self.export_format_combo)
        
        # Data retention
        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(1, 365)
        self.retention_spin.setValue(30)
        self.retention_spin.setSuffix(" days")
        layout.addRow("Data Retention:", self.retention_spin)
        
        # Backup settings
        self.backup_check = QCheckBox("Enable Automatic Backup")
        layout.addRow(self.backup_check)
        
        self.backup_interval_combo = QComboBox()
        self.backup_interval_combo.addItems(["Daily", "Weekly", "Monthly"])
        layout.addRow("Backup Interval:", self.backup_interval_combo)
        
        # Database cleanup
        cleanup_btn = QPushButton(FinaleIcons.get_icon("delete"), "Cleanup Old Data")
        layout.addRow(cleanup_btn)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())
    
    @Slot()
    def browse_data_directory(self):
        """Browse for data directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Data Directory", self.data_dir_edit.text()
        )
        if directory:
            self.data_dir_edit.setText(directory)

class SettingsView(QWidget):
    """
    Main settings view with tabbed configuration sections.
    """
    
    settings_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = load_configuration('config.json')
        # Add configuration panel from original
        self.config_panel = ConfigPanel()
        self.settings = QSettings("Finale", "TrafficMonitoring")
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        """Setup the settings view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Header
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Settings")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        
        # Action buttons
        self.reset_btn = QPushButton(FinaleIcons.get_icon("refresh"), "Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        self.export_btn = QPushButton(FinaleIcons.get_icon("export"), "Export Settings")
        self.export_btn.clicked.connect(self.export_settings)
        
        self.import_btn = QPushButton(FinaleIcons.get_icon("import"), "Import Settings")
        self.import_btn.clicked.connect(self.import_settings)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.reset_btn)
        header_layout.addWidget(self.export_btn)
        header_layout.addWidget(self.import_btn)
        
        layout.addLayout(header_layout)
        
        # Settings tabs
        self.tabs = QTabWidget()
        
        # Create configuration widgets
        self.model_config = ModelConfigWidget()
        self.violation_config = ViolationConfigWidget()
        self.ui_preferences = UIPreferencesWidget()
        self.performance_config = PerformanceWidget()
        self.data_management = DataManagementWidget()
        
        # Add tabs
        self.tabs.addTab(self.model_config, FinaleIcons.get_icon("model"), "AI Models")
        self.tabs.addTab(self.violation_config, FinaleIcons.get_icon("warning"), "Violations")
        self.tabs.addTab(self.ui_preferences, FinaleIcons.get_icon("palette"), "Interface")
        self.tabs.addTab(self.performance_config, FinaleIcons.get_icon("speed"), "Performance")
        self.tabs.addTab(self.data_management, FinaleIcons.get_icon("database"), "Data")
        
        # Style tabs
        self.tabs.setStyleSheet(FinaleStyles.get_tab_widget_style())
        
        layout.addWidget(self.tabs, 1)
        
        # Bottom action bar
        action_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton(FinaleIcons.get_icon("check"), "Apply")
        self.apply_btn.clicked.connect(self.apply_settings)
        
        self.save_btn = QPushButton(FinaleIcons.get_icon("save"), "Save")
        self.save_btn.clicked.connect(self.save_settings)
        
        self.cancel_btn = QPushButton(FinaleIcons.get_icon("close"), "Cancel")
        self.cancel_btn.clicked.connect(self.cancel_changes)
        
        action_layout.addStretch()
        action_layout.addWidget(self.apply_btn)
        action_layout.addWidget(self.save_btn)
        action_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(action_layout)
        
        # Connect signals
        self.ui_preferences.theme_changed.connect(self.on_theme_changed)
        
        # Apply theme
        self.apply_theme(True)
    
    def load_settings(self):
        """Load settings from QSettings"""
        # Load model configuration
        model_config = {
            'vehicle_model': self.settings.value('model/vehicle_model', ''),
            'traffic_model': self.settings.value('model/traffic_model', ''),
            'confidence_threshold': self.settings.value('model/confidence_threshold', 0.3, float),
            'nms_threshold': self.settings.value('model/nms_threshold', 0.45, float),
            'max_detections': self.settings.value('model/max_detections', 100, int),
            'device': self.settings.value('model/device', 'CPU'),
            'optimize_model': self.settings.value('model/optimize_model', True, bool)
        }
        self.model_config.set_config(model_config)
        
        # Load UI preferences
        dark_mode = self.settings.value('ui/dark_mode', True, bool)
        if dark_mode:
            self.ui_preferences.dark_radio.setChecked(True)
        else:
            self.ui_preferences.light_radio.setChecked(True)
    
    @Slot()
    def apply_settings(self):
        """Apply current settings"""
        settings_data = self.get_all_settings()
        self.settings_changed.emit(settings_data)
        
    @Slot()
    def save_settings(self):
        """Save settings to QSettings"""
        # Save model configuration
        model_config = self.model_config.get_config()
        for key, value in model_config.items():
            self.settings.setValue(f'model/{key}', value)
        
        # Save UI preferences
        self.settings.setValue('ui/dark_mode', self.ui_preferences.dark_radio.isChecked())
        
        # Sync settings
        self.settings.sync()
        
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully.")
        save_configuration(settings_data, 'config.json')
    
    @Slot()
    def cancel_changes(self):
        """Cancel changes and reload settings"""
        self.load_settings()
    
    @Slot()
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.settings.clear()
            self.load_settings()
    
    @Slot()
    def export_settings(self):
        """Export settings to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Settings", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            settings_data = self.get_all_settings()
            try:
                with open(file_path, 'w') as f:
                    json.dump(settings_data, f, indent=2)
                QMessageBox.information(self, "Export Successful", "Settings exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export settings:\n{str(e)}")
    
    @Slot()
    def import_settings(self):
        """Import settings from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Settings", "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings_data = json.load(f)
                
                # Apply imported settings
                self.apply_imported_settings(settings_data)
                QMessageBox.information(self, "Import Successful", "Settings imported successfully.")
                
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import settings:\n{str(e)}")
    
    def get_all_settings(self):
        """Get all current settings as dictionary"""
        return {
            'model': self.model_config.get_config(),
            'ui': {
                'dark_mode': self.ui_preferences.dark_radio.isChecked(),
                'language': self.ui_preferences.language_combo.currentText(),
                'font_size': self.ui_preferences.font_size_spin.value(),
                'animations': self.ui_preferences.animations_check.isChecked(),
                'sound': self.ui_preferences.sound_check.isChecked()
            }
        }
    
    def apply_imported_settings(self, settings_data):
        """Apply imported settings data"""
        if 'model' in settings_data:
            self.model_config.set_config(settings_data['model'])
        
        if 'ui' in settings_data:
            ui_settings = settings_data['ui']
            if 'dark_mode' in ui_settings:
                if ui_settings['dark_mode']:
                    self.ui_preferences.dark_radio.setChecked(True)
                else:
                    self.ui_preferences.light_radio.setChecked(True)
    
    @Slot(bool)
    def on_theme_changed(self, dark_mode):
        """Handle theme change"""
        self.apply_theme(dark_mode)
    
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
                    border-radius: 6px;
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: {MaterialColors.primary_variant};
                }}
            """)
    
    def display_timestamp(self, ts):
        return format_timestamp(ts)
    def display_duration(self, seconds):
        return format_duration(seconds)
