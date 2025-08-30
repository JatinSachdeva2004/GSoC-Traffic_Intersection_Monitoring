"""
Analytics View - Traffic analytics and reporting
Displays charts, statistics, and historical data.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QFrame, QScrollArea, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateEdit,
    QComboBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QDate
from PySide6.QtGui import QPixmap, QPainter, QBrush, QColor, QFont

from datetime import datetime, timedelta
import json

# Import finale components
try:
    # Try relative imports first (when running as a package)
    from ..styles import FinaleStyles, MaterialColors
    from ..icons import FinaleIcons
    # Import advanced chart components from original analytics_tab
    import sys
    import os
    from pathlib import Path
    
    # Add parent directory to path to import from qt_app_pyside
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from qt_app_pyside.ui.analytics_tab import ChartWidget, TimeSeriesChart, DetectionPieChart, ViolationBarChart
    from qt_app_pyside.controllers.analytics_controller import AnalyticsController
    from qt_app_pyside.utils.helpers import load_configuration, format_timestamp, format_duration
except ImportError:
    # Fallback for direct execution
    try:
        from styles import FinaleStyles, MaterialColors
        from icons import FinaleIcons
        # Create simplified chart widgets if advanced ones not available
    except ImportError:
        print("Error importing analytics components")
    class ChartWidget(QWidget):
        def __init__(self, title="Chart"):
            super().__init__()
            self.title = title
            self.data = []
            self.chart_type = "line"  # line, bar, pie
            self.setMinimumSize(400, 300)
            
        def set_data(self, data, chart_type="line"):
            """Set chart data and type"""
            self.data = data
            self.chart_type = chart_type
            self.update()
        
        def paintEvent(self, event):
            """Paint the chart"""
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Background
            painter.fillRect(self.rect(), QColor(MaterialColors.surface))
            
            # Border
            painter.setPen(QColor(MaterialColors.outline))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
            
            # Title
            painter.setPen(QColor(MaterialColors.text_primary))
            painter.setFont(QFont("Segoe UI", 12, QFont.Bold))
            title_rect = self.rect().adjusted(10, 10, -10, -10)
            painter.drawText(title_rect, Qt.AlignTop | Qt.AlignLeft, self.title)
            
            # Chart area
            chart_rect = self.rect().adjusted(50, 50, -20, -50)
            
            if not self.data:
                # No data message
                painter.setPen(QColor(MaterialColors.text_secondary))
                painter.setFont(QFont("Segoe UI", 10))
                painter.drawText(chart_rect, Qt.AlignCenter, "No data available")
                return
            
            # Draw chart based on type
            if self.chart_type == "line":
                self.draw_line_chart(painter, chart_rect)
            elif self.chart_type == "bar":
                self.draw_bar_chart(painter, chart_rect)
            elif self.chart_type == "pie":
                self.draw_pie_chart(painter, chart_rect)
        
        def draw_line_chart(self, painter, rect):
            """Draw a line chart"""
            if len(self.data) < 2:
                return
                
            # Find min/max values
            values = [item.get('value', 0) for item in self.data]
            min_val, max_val = min(values), max(values)
            
            if max_val == min_val:
                max_val = min_val + 1
            
            # Calculate points
            points = []
            for i, item in enumerate(self.data):
                x = rect.left() + (i / (len(self.data) - 1)) * rect.width()
                y = rect.bottom() - ((item.get('value', 0) - min_val) / (max_val - min_val)) * rect.height()
                points.append((x, y))
            
            # Draw grid lines
            painter.setPen(QColor(MaterialColors.outline_variant))
            for i in range(5):
                y = rect.top() + (i / 4) * rect.height()
                painter.drawLine(rect.left(), y, rect.right(), y)
            
            # Draw line
            painter.setPen(QColor(MaterialColors.primary))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
            
            # Draw points
            painter.setBrush(QBrush(QColor(MaterialColors.primary)))
            for x, y in points:
                painter.drawEllipse(x-3, y-3, 6, 6)
        
        def draw_bar_chart(self, painter, rect):
            """Draw a bar chart"""
            if not self.data:
                return
                
            values = [item.get('value', 0) for item in self.data]
            max_val = max(values) if values else 1
            
            bar_width = rect.width() / len(self.data) * 0.8
            spacing = rect.width() / len(self.data) * 0.2
            
            painter.setBrush(QBrush(QColor(MaterialColors.primary)))
            
            for i, item in enumerate(self.data):
                value = item.get('value', 0)
                height = (value / max_val) * rect.height()
                
                x = rect.left() + i * (bar_width + spacing) + spacing / 2
                y = rect.bottom() - height
                
                painter.drawRect(x, y, bar_width, height)
        
        def draw_pie_chart(self, painter, rect):
            """Draw a pie chart"""
            if not self.data:
                return
                
            total = sum(item.get('value', 0) for item in self.data)
            if total == 0:
                return
            
            # Calculate center and radius
            center = rect.center()
            radius = min(rect.width(), rect.height()) // 2 - 20
            
            # Colors for pie slices
            colors = [MaterialColors.primary, MaterialColors.secondary, MaterialColors.tertiary,
                     MaterialColors.error, MaterialColors.success, MaterialColors.warning]
            
            start_angle = 0
            for i, item in enumerate(self.data):
                value = item.get('value', 0)
                angle = (value / total) * 360 * 16  # Qt uses 16ths of a degree
                
                color = QColor(colors[i % len(colors)])
                painter.setBrush(QBrush(color))
                painter.setPen(QColor(MaterialColors.outline))
                
                painter.drawPie(center.x() - radius, center.y() - radius, 
                              radius * 2, radius * 2, start_angle, angle)
                
                start_angle += angle

class TrafficSummaryWidget(QGroupBox):
    """
    Widget showing traffic summary statistics.
    """
    
    def __init__(self, parent=None):
        super().__init__("Traffic Summary", parent)
        self.setup_ui()
        self.reset_stats()
        
    def setup_ui(self):
        """Setup summary UI"""
        layout = QGridLayout(self)
        
        # Create stat labels
        self.total_vehicles_label = QLabel("0")
        self.total_violations_label = QLabel("0")
        self.avg_speed_label = QLabel("0.0 km/h")
        self.peak_hour_label = QLabel("N/A")
        
        # Style the stat values
        for label in [self.total_vehicles_label, self.total_violations_label, 
                     self.avg_speed_label, self.peak_hour_label]:
            label.setFont(QFont("Segoe UI", 16, QFont.Bold))
            label.setStyleSheet(f"color: {MaterialColors.primary};")
        
        # Add to layout
        layout.addWidget(QLabel("Total Vehicles:"), 0, 0)
        layout.addWidget(self.total_vehicles_label, 0, 1)
        
        layout.addWidget(QLabel("Total Violations:"), 1, 0)
        layout.addWidget(self.total_violations_label, 1, 1)
        
        layout.addWidget(QLabel("Average Speed:"), 2, 0)
        layout.addWidget(self.avg_speed_label, 2, 1)
        
        layout.addWidget(QLabel("Peak Hour:"), 3, 0)
        layout.addWidget(self.peak_hour_label, 3, 1)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_group_box_style())
    
    def reset_stats(self):
        """Reset all statistics"""
        self.total_vehicles_label.setText("0")
        self.total_violations_label.setText("0")
        self.avg_speed_label.setText("0.0 km/h")
        self.peak_hour_label.setText("N/A")
    
    def update_stats(self, stats):
        """Update statistics display"""
        if 'total_vehicles' in stats:
            self.total_vehicles_label.setText(str(stats['total_vehicles']))
        
        if 'total_violations' in stats:
            self.total_violations_label.setText(str(stats['total_violations']))
        
        if 'avg_speed' in stats:
            self.avg_speed_label.setText(f"{stats['avg_speed']:.1f} km/h")
        
        if 'peak_hour' in stats:
            self.peak_hour_label.setText(stats['peak_hour'])

class ViolationsTableWidget(QTableWidget):
    """
    Table widget for displaying violation records.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_table()
        
    def setup_table(self):
        """Setup the violations table"""
        # Set columns
        columns = ["Time", "Type", "Vehicle", "Location", "Confidence", "Actions"]
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        # Configure table
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setAlternatingRowColors(True)
        
        # Apply styling
        self.setStyleSheet(FinaleStyles.get_table_style())
    
    def add_violation(self, violation_data):
        """Add a violation record to the table"""
        row = self.rowCount()
        self.insertRow(row)
        
        # Populate row data
        time_str = violation_data.get('timestamp', datetime.now().strftime('%H:%M:%S'))
        violation_type = violation_data.get('type', 'Red Light')
        vehicle_id = violation_data.get('vehicle_id', 'Unknown')
        location = violation_data.get('location', 'Intersection 1')
        confidence = violation_data.get('confidence', 0.0)
        
        self.setItem(row, 0, QTableWidgetItem(time_str))
        self.setItem(row, 1, QTableWidgetItem(violation_type))
        self.setItem(row, 2, QTableWidgetItem(vehicle_id))
        self.setItem(row, 3, QTableWidgetItem(location))
        self.setItem(row, 4, QTableWidgetItem(f"{confidence:.2f}"))
        
        # Actions button
        actions_btn = QPushButton("View Details")
        actions_btn.clicked.connect(lambda: self.view_violation_details(violation_data))
        self.setCellWidget(row, 5, actions_btn)
        
        # Auto-scroll to new violation
        self.scrollToBottom()
    
    def view_violation_details(self, violation_data):
        """View detailed violation information"""
        # This could open a detailed dialog
        print(f"Viewing violation details: {violation_data}")

class AnalyticsView(QWidget):
    """
    Main analytics view with charts, statistics, and violation history.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analytics_controller = AnalyticsController()
        self.setup_ui()
        self.analytics_controller.data_updated.connect(self.refresh_analytics)
        # Load config if needed
        self.config = load_configuration('config.json')

    def setup_ui(self):
        """Setup the analytics view UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Date range selection
        controls_layout.addWidget(QLabel("Date Range:"))
        
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.start_date.setCalendarPopup(True)
        controls_layout.addWidget(self.start_date)
        
        controls_layout.addWidget(QLabel("to"))
        
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)
        controls_layout.addWidget(self.end_date)
        
        # Time interval
        controls_layout.addWidget(QLabel("Interval:"))
        self.interval_combo = QComboBox()
        self.interval_combo.addItems(["Hourly", "Daily", "Weekly"])
        controls_layout.addWidget(self.interval_combo)
        
        # Refresh button
        self.refresh_btn = QPushButton(FinaleIcons.get_icon("refresh"), "Refresh")
        self.refresh_btn.clicked.connect(self.refresh_data)
        controls_layout.addWidget(self.refresh_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left panel - Charts
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        
        # Traffic flow chart
        self.traffic_chart = AnalyticsChartWidget("Traffic Flow Over Time")
        charts_layout.addWidget(self.traffic_chart)
        
        # Violation types chart
        self.violations_chart = AnalyticsChartWidget("Violation Types")
        charts_layout.addWidget(self.violations_chart)
        
        content_layout.addWidget(charts_widget, 2)
        
        # Right panel - Statistics and table
        right_panel = QVBoxLayout()
        
        # Summary statistics
        self.summary_widget = TrafficSummaryWidget()
        right_panel.addWidget(self.summary_widget)
        
        # Recent violations table
        violations_group = QGroupBox("Recent Violations")
        violations_layout = QVBoxLayout(violations_group)
        
        self.violations_table = ViolationsTableWidget()
        violations_layout.addWidget(self.violations_table)
        
        violations_group.setStyleSheet(FinaleStyles.get_group_box_style())
        right_panel.addWidget(violations_group, 1)
        
        content_layout.addLayout(right_panel, 1)
        layout.addLayout(content_layout, 1)
        
        # Apply theme
        self.apply_theme(True)
        
        # Load initial data
        self.refresh_data()
    
    @Slot()
    def refresh_data(self):
        """Refresh analytics data"""
        print("Refreshing analytics data...")
        
        # Update traffic flow chart (sample data)
        traffic_data = [
            {'label': '08:00', 'value': 45},
            {'label': '09:00', 'value': 67},
            {'label': '10:00', 'value': 89},
            {'label': '11:00', 'value': 76},
            {'label': '12:00', 'value': 92},
            {'label': '13:00', 'value': 84},
            {'label': '14:00', 'value': 71}
        ]
        self.traffic_chart.set_data(traffic_data, "line")
        
        # Update violations chart
        violations_data = [
            {'label': 'Red Light', 'value': 12},
            {'label': 'Speed', 'value': 8},
            {'label': 'Wrong Lane', 'value': 5},
            {'label': 'No Helmet', 'value': 3}
        ]
        self.violations_chart.set_data(violations_data, "pie")
        
        # Update summary
        summary_stats = {
            'total_vehicles': 1247,
            'total_violations': 28,
            'avg_speed': 35.2,
            'peak_hour': '12:00-13:00'
        }
        self.summary_widget.update_stats(summary_stats)
    
    def refresh_analytics(self):
        """Refresh analytics data from controller"""
        data = self.analytics_controller.get_analytics_data()
        # Use format_timestamp, format_duration for display
        # ... update charts and stats with new data ...
    
    def update_demo_data(self):
        """Update with demo data for demonstration"""
        import random
        
        # Simulate new violation
        if random.random() < 0.3:  # 30% chance
            violation = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'type': random.choice(['Red Light', 'Speed', 'Wrong Lane']),
                'vehicle_id': f"VH{random.randint(1000, 9999)}",
                'location': f"Intersection {random.randint(1, 5)}",
                'confidence': random.uniform(0.7, 0.95)
            }
            self.violations_table.add_violation(violation)
    
    def add_violation(self, violation_data):
        """Add a new violation (called from main window)"""
        self.violations_table.add_violation(violation_data)
    
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
                QDateEdit, QComboBox {{
                    background-color: {MaterialColors.surface_variant};
                    border: 1px solid {MaterialColors.outline};
                    border-radius: 4px;
                    padding: 6px;
                }}
            """)
