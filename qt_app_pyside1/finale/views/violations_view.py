"""
Violations View - Violation management and history
Displays violation records, details, and management tools.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFrame, QScrollArea, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
    QComboBox, QSpinBox, QLineEdit, QTextEdit, QDialog,
    QDialogButtonBox, QSplitter, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QDate, QSize
from PySide6.QtGui import QPixmap, QPainter, QBrush, QColor, QFont, QIcon

from datetime import datetime, timedelta
import json
import os

# Import finale components
from ..styles import FinaleStyles, MaterialColors
from ..icons import FinaleIcons
from qt_app_pyside.utils.helpers import save_configuration, create_export_csv, create_export_json
from qt_app_pyside.utils.annotation_utils import draw_detections
from qt_app_pyside.utils.enhanced_annotation_utils import enhanced_draw_detections
from qt_app_pyside.ui.export_tab import ExportTab
from qt_app_pyside.ui.violations_tab import ViolationsTab as OriginalViolationsTab

class ViolationDetailDialog(QDialog):
    """
    Dialog for viewing detailed violation information.
    """
    
    def __init__(self, violation_data, parent=None):
        super().__init__(parent)
        self.violation_data = violation_data
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the detail dialog UI"""
        self.setWindowTitle("Violation Details")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Header with violation type and timestamp
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {MaterialColors.primary};
                color: {MaterialColors.text_on_primary};
                border-radius: 8px;
                padding: 16px;
            }}
        """)
        
        header_layout = QHBoxLayout(header_frame)
        
        violation_type = self.violation_data.get('type', 'Unknown')
        timestamp = self.violation_data.get('timestamp', 'Unknown')
        
        type_label = QLabel(violation_type)
        type_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        
        time_label = QLabel(timestamp)
        time_label.setFont(QFont("Segoe UI", 12))
        
        header_layout.addWidget(type_label)
        header_layout.addStretch()
        header_layout.addWidget(time_label)
        
        layout.addWidget(header_frame)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Image/Video
        image_group = QGroupBox("Evidence")
        image_layout = QVBoxLayout(image_group)
        
        self.image_label = QLabel()
        self.image_label.setMinimumSize(300, 200)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #424242;
                border-radius: 8px;
                background-color: #1a1a1a;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("No image available")
        
        # Load image if available
        image_path = self.violation_data.get('image_path')
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
        
        image_layout.addWidget(self.image_label)
        
        # Image controls
        image_controls = QHBoxLayout()
        
        save_image_btn = QPushButton(FinaleIcons.get_icon("save"), "Save Image")
        view_full_btn = QPushButton(FinaleIcons.get_icon("zoom_in"), "View Full")
        
        image_controls.addWidget(save_image_btn)
        image_controls.addWidget(view_full_btn)
        image_controls.addStretch()
        
        image_layout.addLayout(image_controls)
        
        content_splitter.addWidget(image_group)
        
        # Right side - Details
        details_group = QGroupBox("Details")
        details_layout = QGridLayout(details_group)
        
        # Violation details
        details = [
            ("Vehicle ID:", self.violation_data.get('vehicle_id', 'Unknown')),
            ("Location:", self.violation_data.get('location', 'Unknown')),
            ("Confidence:", f"{self.violation_data.get('confidence', 0.0):.2f}"),
            ("Speed:", f"{self.violation_data.get('speed', 0.0):.1f} km/h"),
            ("Lane:", self.violation_data.get('lane', 'Unknown')),
            ("Weather:", self.violation_data.get('weather', 'Unknown')),
            ("Officer ID:", self.violation_data.get('officer_id', 'N/A')),
            ("Status:", self.violation_data.get('status', 'Pending'))
        ]
        
        for i, (label, value) in enumerate(details):
            label_widget = QLabel(label)
            label_widget.setFont(QFont("Segoe UI", 9, QFont.Bold))
            
            value_widget = QLabel(str(value))
            value_widget.setStyleSheet(f"color: {MaterialColors.text_secondary};")
            
            details_layout.addWidget(label_widget, i, 0)
            details_layout.addWidget(value_widget, i, 1)
        
        # Notes section
        notes_label = QLabel("Notes:")
        notes_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        details_layout.addWidget(notes_label, len(details), 0, 1, 2)
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        self.notes_edit.setPlainText(self.violation_data.get('notes', ''))
        details_layout.addWidget(self.notes_edit, len(details) + 1, 0, 1, 2)
        
        content_splitter.addWidget(details_group)
        layout.addWidget(content_splitter)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton(FinaleIcons.get_icon("export"), "Export Report")
        delete_btn = QPushButton(FinaleIcons.get_icon("delete"), "Delete")
        delete_btn.setStyleSheet(f"background-color: {MaterialColors.error};")
        
        button_layout.addWidget(export_btn)
        button_layout.addWidget(delete_btn)
        button_layout.addStretch()
        
        # Standard dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        button_box.accepted.connect(self.save_changes)
        button_box.rejected.connect(self.reject)
        
        button_layout.addWidget(button_box)
        layout.addLayout(button_layout)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_dialog_style())
    
    @Slot()
    def save_changes(self):
        """Save changes to violation data"""
        # Update notes
        self.violation_data['notes'] = self.notes_edit.toPlainText()
        # Here you would save to database/file
        self.accept()

class ViolationFilterWidget(QGroupBox):
    """
    Widget for filtering violations by various criteria.
    """
    
    filter_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__("Filter Violations", parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup filter UI"""
        layout = QGridLayout(self)
        
        # Date range
        layout.addWidget(QLabel("Date From:"), 0, 0)
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        layout.addWidget(self.date_from, 0, 1)
        
        layout.addWidget(QLabel("Date To:"), 0, 2)
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        layout.addWidget(self.date_to, 0, 3)
        
        # Violation type
        layout.addWidget(QLabel("Type:"), 1, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["All Types", "Red Light", "Speed", "Wrong Lane", "No Helmet", "Other"])
        layout.addWidget(self.type_combo, 1, 1)
        
        # Status
        layout.addWidget(QLabel("Status:"), 1, 2)
        self.status_combo = QComboBox()
        self.status_combo.addItems(["All Status", "Pending", "Reviewed", "Closed", "Disputed"])
        layout.addWidget(self.status_combo, 1, 3)
        
        # Location
        layout.addWidget(QLabel("Location:"), 2, 0)
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("Enter location...")
        layout.addWidget(self.location_edit, 2, 1)
        
        # Confidence threshold
        layout.addWidget(QLabel("Min Confidence:"), 2, 2)
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(50)
        self.confidence_spin.setSuffix("%")
        layout.addWidget(self.confidence_spin, 2, 3)
        
        # Apply button
        self.apply_btn = QPushButton(FinaleIcons.get_icon("filter"), "Apply Filter")
        self.apply_btn.clicked.connect(self.apply_filter)
        layout.addWidget(self.apply_btn, 3, 0, 1, 4)
        
        # Connect signals for auto-update
        self.date_from.dateChanged.connect(self.on_filter_changed)
        self.date_to.dateChanged.connect(self.on_filter_changed)
        self.type_combo.currentTextChanged.connect(self.on_filter_changed)
        self.status_combo.currentTextChanged.connect(self.on_filter_changed)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())
    
    @Slot()
    def apply_filter(self):
        """Apply current filter settings"""
        self.on_filter_changed()
    
    def on_filter_changed(self):
        """Emit filter changed signal with current settings"""
        filter_data = {
            'date_from': self.date_from.date().toPython(),
            'date_to': self.date_to.date().toPython(),
            'type': self.type_combo.currentText(),
            'status': self.status_combo.currentText(),
            'location': self.location_edit.text(),
            'min_confidence': self.confidence_spin.value() / 100.0
        }
        self.filter_changed.emit(filter_data)

class ViolationListWidget(QWidget):
    """
    Widget displaying violation list with thumbnails and quick info.
    """
    
    violation_selected = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.violations = []
        self.setup_ui()
        
    def setup_ui(self):
        """Setup violation list UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_layout = QHBoxLayout()
        
        self.count_label = QLabel("0 violations")
        self.count_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Sort by Time", "Sort by Type", "Sort by Confidence", "Sort by Status"])
        self.sort_combo.currentTextChanged.connect(self.sort_violations)
        
        header_layout.addWidget(self.count_label)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Sort:"))
        header_layout.addWidget(self.sort_combo)
        
        layout.addLayout(header_layout)
        
        # Violations list
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.list_widget.setStyleSheet(FinaleStyles.get_list_style())
        
        layout.addWidget(self.list_widget)
        
    def add_violation(self, violation_data):
        """Add a violation to the list"""
        self.violations.append(violation_data)
        self.update_list()
        
    def set_violations(self, violations):
        """Set the complete list of violations"""
        self.violations = violations
        self.update_list()
        
    def update_list(self):
        """Update the violation list display"""
        self.list_widget.clear()
        
        for violation in self.violations:
            item = QListWidgetItem()
            
            # Create custom widget for violation item
            item_widget = self.create_violation_item_widget(violation)
            
            item.setSizeHint(item_widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, item_widget)
        
        # Update count
        self.count_label.setText(f"{len(self.violations)} violations")
    
    def create_violation_item_widget(self, violation):
        """Create a custom widget for a violation list item"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Thumbnail (placeholder for now)
        thumbnail = QLabel()
        thumbnail.setFixedSize(80, 60)
        thumbnail.setStyleSheet("""
            QLabel {
                border: 1px solid #424242;
                border-radius: 4px;
                background-color: #2d2d2d;
            }
        """)
        thumbnail.setAlignment(Qt.AlignCenter)
        thumbnail.setText("IMG")
        layout.addWidget(thumbnail)
        
        # Violation info
        info_layout = QVBoxLayout()
        
        # Title line
        title_layout = QHBoxLayout()
        
        type_label = QLabel(violation.get('type', 'Unknown'))
        type_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        
        time_label = QLabel(violation.get('timestamp', ''))
        time_label.setStyleSheet(f"color: {MaterialColors.text_secondary}; font-size: 10px;")
        
        title_layout.addWidget(type_label)
        title_layout.addStretch()
        title_layout.addWidget(time_label)
        
        info_layout.addLayout(title_layout)
        
        # Details line
        details = f"Vehicle: {violation.get('vehicle_id', 'Unknown')} | Location: {violation.get('location', 'Unknown')}"
        details_label = QLabel(details)
        details_label.setStyleSheet(f"color: {MaterialColors.text_secondary}; font-size: 9px;")
        info_layout.addWidget(details_label)
        
        # Confidence and status
        status_layout = QHBoxLayout()
        
        confidence = violation.get('confidence', 0.0)
        confidence_label = QLabel(f"Confidence: {confidence:.2f}")
        confidence_label.setStyleSheet(f"color: {MaterialColors.primary}; font-size: 9px;")
        
        status = violation.get('status', 'Pending')
        status_label = QLabel(status)
        status_color = {
            'Pending': MaterialColors.warning,
            'Reviewed': MaterialColors.primary,
            'Closed': MaterialColors.success,
            'Disputed': MaterialColors.error
        }.get(status, MaterialColors.text_secondary)
        status_label.setStyleSheet(f"color: {status_color}; font-size: 9px; font-weight: bold;")
        
        status_layout.addWidget(confidence_label)
        status_layout.addStretch()
        status_layout.addWidget(status_label)
        
        info_layout.addLayout(status_layout)
        layout.addLayout(info_layout, 1)
        
        # Store violation data in widget
        widget.violation_data = violation
        
        return widget
    
    def sort_violations(self, sort_by):
        """Sort violations by the specified criteria"""
        if sort_by == "Sort by Time":
            self.violations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        elif sort_by == "Sort by Type":
            self.violations.sort(key=lambda x: x.get('type', ''))
        elif sort_by == "Sort by Confidence":
            self.violations.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        elif sort_by == "Sort by Status":
            self.violations.sort(key=lambda x: x.get('status', ''))
        
        self.update_list()
    
    @Slot(QListWidgetItem)
    def on_item_clicked(self, item):
        """Handle violation item click"""
        item_widget = self.list_widget.itemWidget(item)
        if hasattr(item_widget, 'violation_data'):
            self.violation_selected.emit(item_widget.violation_data)

class ViolationsView(QWidget):
    """
    Main violations view with filtering, list, and detail management.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.load_sample_data()
        
        self.save_config = save_configuration
        self.export_csv = create_export_csv
        self.export_json = create_export_json
        self.draw_detections = draw_detections
        self.enhanced_draw_detections = enhanced_draw_detections
        # Add export functionality from original export_tab
        self.export_handler = ExportTab()
    
    def setup_ui(self):
        """Setup the violations view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Filter widget
        self.filter_widget = ViolationFilterWidget()
        self.filter_widget.filter_changed.connect(self.apply_filter)
        layout.addWidget(self.filter_widget)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Violation list
        self.violation_list = ViolationListWidget()
        self.violation_list.violation_selected.connect(self.show_violation_details)
        content_splitter.addWidget(self.violation_list)
        
        # Right side - Quick actions and summary
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Quick actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        export_all_btn = QPushButton(FinaleIcons.get_icon("export"), "Export All")
        export_filtered_btn = QPushButton(FinaleIcons.get_icon("filter"), "Export Filtered")
        delete_selected_btn = QPushButton(FinaleIcons.get_icon("delete"), "Delete Selected")
        mark_reviewed_btn = QPushButton(FinaleIcons.get_icon("check"), "Mark as Reviewed")
        
        actions_layout.addWidget(export_all_btn)
        actions_layout.addWidget(export_filtered_btn)
        actions_layout.addWidget(delete_selected_btn)
        actions_layout.addWidget(mark_reviewed_btn)
        
        actions_group.setStyleSheet(FinaleStyles.get_group_box_style())
        right_layout.addWidget(actions_group)
        
        # Summary statistics
        summary_group = QGroupBox("Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.total_label = QLabel("Total: 0")
        self.pending_label = QLabel("Pending: 0")
        self.reviewed_label = QLabel("Reviewed: 0")
        self.closed_label = QLabel("Closed: 0")
        
        summary_layout.addWidget(self.total_label, 0, 0)
        summary_layout.addWidget(self.pending_label, 0, 1)
        summary_layout.addWidget(self.reviewed_label, 1, 0)
        summary_layout.addWidget(self.closed_label, 1, 1)
        
        summary_group.setStyleSheet(FinaleStyles.get_group_box_style())
        right_layout.addWidget(summary_group)
        
        right_layout.addStretch()
        content_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        content_splitter.setSizes([700, 300])
        
        layout.addWidget(content_splitter, 1)
        
        # Apply theme
        self.apply_theme(True)
    
    def load_sample_data(self):
        """Load sample violation data for demonstration"""
        sample_violations = [
            {
                'timestamp': '14:23:15',
                'type': 'Red Light',
                'vehicle_id': 'VH1234',
                'location': 'Main St & 1st Ave',
                'confidence': 0.92,
                'status': 'Pending',
                'speed': 45.2,
                'lane': 'Left Turn',
                'notes': 'Clear violation captured on camera.'
            },
            {
                'timestamp': '13:45:32',
                'type': 'Speed',
                'vehicle_id': 'VH5678',
                'location': 'Highway 101',
                'confidence': 0.87,
                'status': 'Reviewed',
                'speed': 78.5,
                'lane': 'Right',
                'notes': 'Speed limit 60 km/h, vehicle traveling at 78.5 km/h.'
            },
            {
                'timestamp': '12:15:48',
                'type': 'Wrong Lane',
                'vehicle_id': 'VH9012',
                'location': 'Oak St Bridge',
                'confidence': 0.76,
                'status': 'Closed',
                'speed': 32.1,
                'lane': 'Bus Lane',
                'notes': 'Vehicle in bus-only lane during restricted hours.'
            }
        ]
        
        self.violation_list.set_violations(sample_violations)
        self.update_summary()
    
    def add_violation(self, violation_data):
        """Add a new violation (called from main window)"""
        self.violation_list.add_violation(violation_data)
        self.update_summary()
    
    @Slot(dict)
    def apply_filter(self, filter_data):
        """Apply filter to violation list"""
        print(f"Applying filter: {filter_data}")
        # Here you would filter the violations based on criteria
        # For now, just update summary
        self.update_summary()
    
    @Slot(dict)
    def show_violation_details(self, violation_data):
        """Show detailed view of selected violation"""
        dialog = ViolationDetailDialog(violation_data, self)
        dialog.exec()
    
    def update_summary(self):
        """Update summary statistics"""
        violations = self.violation_list.violations
        
        total = len(violations)
        pending = len([v for v in violations if v.get('status') == 'Pending'])
        reviewed = len([v for v in violations if v.get('status') == 'Reviewed'])
        closed = len([v for v in violations if v.get('status') == 'Closed'])
        
        self.total_label.setText(f"Total: {total}")
        self.pending_label.setText(f"Pending: {pending}")
        self.reviewed_label.setText(f"Reviewed: {reviewed}")
        self.closed_label.setText(f"Closed: {closed}")
    
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
