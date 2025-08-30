"""
Modern Dark Theme and Styling System
===================================

Complete styling system with Material Design 3.0 principles, dark theme,
animations, and responsive design for the Traffic Monitoring Application.

Features:
- Material Design 3.0 dark theme
- Animated transitions and hover effects
- Responsive typography and spacing
- Custom widget styling
- Accent color system
- Professional gradients and shadows
"""

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QLinearGradient, QBrush
from PySide6.QtWidgets import QApplication, QWidget
from typing import Dict, Optional
import json

class Colors:
    """Material Design 3.0 Color Palette - Dark Theme"""
    
    # Primary colors
    PRIMARY_BACKGROUND = "#121212"
    SECONDARY_BACKGROUND = "#1E1E1E"
    SURFACE = "#2C2C2C"
    SURFACE_VARIANT = "#383838"
    
    # Accent colors
    ACCENT_CYAN = "#00BCD4"
    ACCENT_GREEN = "#4CAF50"
    ACCENT_RED = "#FF5722"
    ACCENT_YELLOW = "#FFC107"
    ACCENT_BLUE = "#2196F3"
    ACCENT_PURPLE = "#9C27B0"
    
    # Text colors
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#B0B0B0"
    TEXT_DISABLED = "#757575"
    
    # State colors
    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"
    ERROR = "#F44336"
    INFO = "#2196F3"
    
    # Border and divider
    BORDER = "#424242"
    DIVIDER = "#2C2C2C"
    
    # Interactive states
    HOVER = "#404040"
    PRESSED = "#505050"
    SELECTED = "#1976D2"
    FOCUS = "#03DAC6"

class Fonts:
    """Typography system with hierarchy"""
    
    @staticmethod
    def get_font(size: int = 10, weight: str = "normal", family: str = "Segoe UI") -> QFont:
        """Get a font with specified parameters"""
        font = QFont(family, size)
        
        weight_map = {
            "light": QFont.Weight.Light,
            "normal": QFont.Weight.Normal,
            "medium": QFont.Weight.Medium,
            "semibold": QFont.Weight.DemiBold,
            "bold": QFont.Weight.Bold
        }
        
        font.setWeight(weight_map.get(weight, QFont.Weight.Normal))
        return font
    
    @staticmethod
    def heading_1() -> QFont:
        return Fonts.get_font(24, "bold")
    
    @staticmethod
    def heading_2() -> QFont:
        return Fonts.get_font(20, "semibold")
    
    @staticmethod
    def heading_3() -> QFont:
        return Fonts.get_font(16, "semibold")
    
    @staticmethod
    def body_large() -> QFont:
        return Fonts.get_font(14, "normal")
    
    @staticmethod
    def body_medium() -> QFont:
        return Fonts.get_font(12, "normal")
    
    @staticmethod
    def body_small() -> QFont:
        return Fonts.get_font(10, "normal")
    
    @staticmethod
    def caption() -> QFont:
        return Fonts.get_font(9, "normal")
    
    @staticmethod
    def button() -> QFont:
        return Fonts.get_font(12, "medium")

class Spacing:
    """Consistent spacing system"""
    XS = 4
    SM = 8
    MD = 16
    LG = 24
    XL = 32
    XXL = 48

class BorderRadius:
    """Border radius system"""
    SM = 4
    MD = 8
    LG = 12
    XL = 16
    PILL = 9999

class ThemeManager:
    """Manages application theme and styling"""
    
    def __init__(self, accent_color: str = Colors.ACCENT_CYAN):
        self.accent_color = accent_color
        self._setup_palette()
    
    def _setup_palette(self):
        """Setup Qt application palette"""
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(Colors.PRIMARY_BACKGROUND))
        palette.setColor(QPalette.WindowText, QColor(Colors.TEXT_PRIMARY))
        
        # Base colors (input fields)
        palette.setColor(QPalette.Base, QColor(Colors.SURFACE))
        palette.setColor(QPalette.Text, QColor(Colors.TEXT_PRIMARY))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(Colors.SURFACE))
        palette.setColor(QPalette.ButtonText, QColor(Colors.TEXT_PRIMARY))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(self.accent_color))
        palette.setColor(QPalette.HighlightedText, QColor(Colors.TEXT_PRIMARY))
        
        # Apply palette
        if QApplication.instance():
            QApplication.instance().setPalette(palette)
    
    def set_accent_color(self, color: str):
        """Change the accent color"""
        self.accent_color = color
        self._setup_palette()

class StyleSheets:
    """Collection of Qt StyleSheets for various components"""
    
    @staticmethod
    def main_window() -> str:
        return f"""
        QMainWindow {{
            background-color: {Colors.PRIMARY_BACKGROUND};
            color: {Colors.TEXT_PRIMARY};
        }}
        
        QMainWindow::separator {{
            background-color: {Colors.BORDER};
            width: 1px;
            height: 1px;
        }}
        """
    
    @staticmethod
    def tab_widget() -> str:
        return f"""
        QTabWidget::pane {{
            border: 1px solid {Colors.BORDER};
            background-color: {Colors.SECONDARY_BACKGROUND};
            border-radius: {BorderRadius.MD}px;
        }}
        
        QTabBar::tab {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_SECONDARY};
            padding: {Spacing.SM}px {Spacing.MD}px;
            margin-right: 2px;
            border-top-left-radius: {BorderRadius.SM}px;
            border-top-right-radius: {BorderRadius.SM}px;
            font-weight: 500;
            min-width: 100px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {Colors.ACCENT_CYAN};
            color: {Colors.TEXT_PRIMARY};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {Colors.HOVER};
            color: {Colors.TEXT_PRIMARY};
        }}
        """
    
    @staticmethod
    def button_primary() -> str:
        return f"""
        QPushButton {{
            background-color: {Colors.ACCENT_CYAN};
            color: {Colors.TEXT_PRIMARY};
            border: none;
            padding: {Spacing.SM}px {Spacing.MD}px;
            border-radius: {BorderRadius.SM}px;
            font-weight: 500;
            min-height: 32px;
        }}
        
        QPushButton:hover {{
            background-color: #00ACC1;
        }}
        
        QPushButton:pressed {{
            background-color: #0097A7;
        }}
        
        QPushButton:disabled {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_DISABLED};
        }}
        """
    
    @staticmethod
    def button_secondary() -> str:
        return f"""
        QPushButton {{
            background-color: transparent;
            color: {Colors.ACCENT_CYAN};
            border: 2px solid {Colors.ACCENT_CYAN};
            padding: {Spacing.SM}px {Spacing.MD}px;
            border-radius: {BorderRadius.SM}px;
            font-weight: 500;
            min-height: 32px;
        }}
        
        QPushButton:hover {{
            background-color: rgba(0, 188, 212, 0.1);
        }}
        
        QPushButton:pressed {{
            background-color: rgba(0, 188, 212, 0.2);
        }}
        """
    
    @staticmethod
    def card() -> str:
        return f"""
        QWidget {{
            background-color: {Colors.SURFACE};
            border: 1px solid {Colors.BORDER};
            border-radius: {BorderRadius.MD}px;
            padding: {Spacing.MD}px;
        }}
        """
    
    @staticmethod
    def input_field() -> str:
        return f"""
        QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            border: 2px solid {Colors.BORDER};
            border-radius: {BorderRadius.SM}px;
            padding: {Spacing.SM}px;
            font-size: 12px;
        }}
        
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, 
        QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {Colors.ACCENT_CYAN};
        }}
        
        QLineEdit:hover, QTextEdit:hover, QSpinBox:hover,
        QDoubleSpinBox:hover, QComboBox:hover {{
            border-color: {Colors.HOVER};
        }}
        """
    
    @staticmethod
    def table() -> str:
        return f"""
        QTableWidget {{
            background-color: {Colors.SURFACE};
            color: {Colors.TEXT_PRIMARY};
            gridline-color: {Colors.BORDER};
            border: 1px solid {Colors.BORDER};
            border-radius: {BorderRadius.SM}px;
        }}
        
        QTableWidget::item {{
            padding: {Spacing.SM}px;
            border-bottom: 1px solid {Colors.BORDER};
        }}
        
        QTableWidget::item:selected {{
            background-color: {Colors.SELECTED};
        }}
        
        QTableWidget::item:hover {{
            background-color: {Colors.HOVER};
        }}
        
        QHeaderView::section {{
            background-color: {Colors.SURFACE_VARIANT};
            color: {Colors.TEXT_PRIMARY};
            padding: {Spacing.SM}px;
            border: none;
            font-weight: 600;
        }}
        """
    
    @staticmethod
    def scroll_bar() -> str:
        return f"""
        QScrollBar:vertical {{
            background-color: {Colors.SURFACE};
            width: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {Colors.BORDER};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {Colors.HOVER};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        
        QScrollBar:horizontal {{
            background-color: {Colors.SURFACE};
            height: 12px;
            border-radius: 6px;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {Colors.BORDER};
            border-radius: 6px;
            min-width: 20px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {Colors.HOVER};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        """
    
    @staticmethod
    def progress_bar() -> str:
        return f"""
        QProgressBar {{
            background-color: {Colors.SURFACE};
            border: none;
            border-radius: {BorderRadius.SM}px;
            text-align: center;
            height: 8px;
        }}
        
        QProgressBar::chunk {{
            background-color: {Colors.ACCENT_CYAN};
            border-radius: {BorderRadius.SM}px;
        }}
        """
    
    @staticmethod
    def status_bar() -> str:
        return f"""
        QStatusBar {{
            background-color: {Colors.SURFACE_VARIANT};
            color: {Colors.TEXT_SECONDARY};
            border-top: 1px solid {Colors.BORDER};
            padding: {Spacing.SM}px;
        }}
        """
    
    @staticmethod
    def toolbar() -> str:
        return f"""
        QToolBar {{
            background-color: {Colors.SURFACE_VARIANT};
            border: none;
            spacing: {Spacing.SM}px;
            padding: {Spacing.SM}px;
        }}
        
        QToolButton {{
            background-color: transparent;
            color: {Colors.TEXT_PRIMARY};
            border: none;
            border-radius: {BorderRadius.SM}px;
            padding: {Spacing.SM}px;
            min-width: 36px;
            min-height: 36px;
        }}
        
        QToolButton:hover {{
            background-color: {Colors.HOVER};
        }}
        
        QToolButton:pressed {{
            background-color: {Colors.PRESSED};
        }}
        
        QToolButton:checked {{
            background-color: {Colors.ACCENT_CYAN};
        }}
        """
    
    @staticmethod
    def dock_widget() -> str:
        return f"""
        QDockWidget {{
            background-color: {Colors.SECONDARY_BACKGROUND};
            color: {Colors.TEXT_PRIMARY};
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        
        QDockWidget::title {{
            background-color: {Colors.SURFACE_VARIANT};
            padding: {Spacing.SM}px;
            font-weight: 600;
        }}
        """

class AnimationManager:
    """Manages UI animations and transitions"""
    
    @staticmethod
    def create_fade_animation(widget: QWidget, duration: int = 300) -> QPropertyAnimation:
        """Create a fade in/out animation"""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        return animation
    
    @staticmethod
    def create_slide_animation(widget: QWidget, start_pos: QRect, end_pos: QRect, duration: int = 300) -> QPropertyAnimation:
        """Create a slide animation"""
        animation = QPropertyAnimation(widget, b"geometry")
        animation.setDuration(duration)
        animation.setStartValue(start_pos)
        animation.setEndValue(end_pos)
        animation.setEasingCurve(QEasingCurve.OutCubic)
        return animation
    
    @staticmethod
    def pulse_widget(widget: QWidget, duration: int = 1000):
        """Create a pulsing effect on a widget"""
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setKeyValueAt(0.5, 0.5)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.InOutSine)
        animation.setLoopCount(-1)  # Infinite loop
        animation.start()
        return animation

def apply_theme(app: QApplication, theme_manager: Optional[ThemeManager] = None):
    """Apply the complete theme to the application"""
    if not theme_manager:
        theme_manager = ThemeManager()
    
    # Set application style
    app.setStyle("Fusion")
    
    # Apply global stylesheet
    global_style = f"""
    * {{
        font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    }}
    
    {StyleSheets.main_window()}
    {StyleSheets.tab_widget()}
    {StyleSheets.input_field()}
    {StyleSheets.table()}
    {StyleSheets.scroll_bar()}
    {StyleSheets.progress_bar()}
    {StyleSheets.status_bar()}
    {StyleSheets.toolbar()}
    {StyleSheets.dock_widget()}
    
    QWidget {{
        background-color: {Colors.PRIMARY_BACKGROUND};
        color: {Colors.TEXT_PRIMARY};
    }}
    
    QGroupBox {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-radius: {BorderRadius.MD}px;
        margin-top: {Spacing.MD}px;
        padding-top: {Spacing.SM}px;
        font-weight: 600;
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: {Spacing.MD}px;
        padding: 0 {Spacing.SM}px 0 {Spacing.SM}px;
    }}
    
    QCheckBox, QRadioButton {{
        color: {Colors.TEXT_PRIMARY};
        spacing: {Spacing.SM}px;
    }}
    
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 18px;
        height: 18px;
        border: 2px solid {Colors.BORDER};
        border-radius: 4px;
        background-color: {Colors.SURFACE};
    }}
    
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {Colors.ACCENT_CYAN};
        border-color: {Colors.ACCENT_CYAN};
    }}
    
    QSlider::groove:horizontal {{
        height: 6px;
        background-color: {Colors.SURFACE};
        border-radius: 3px;
    }}
    
    QSlider::handle:horizontal {{
        background-color: {Colors.ACCENT_CYAN};
        border: none;
        width: 18px;
        height: 18px;
        border-radius: 9px;
        margin: -6px 0;
    }}
    
    QSlider::sub-page:horizontal {{
        background-color: {Colors.ACCENT_CYAN};
        border-radius: 3px;
    }}
    
    QMenu {{
        background-color: {Colors.SURFACE};
        color: {Colors.TEXT_PRIMARY};
        border: 1px solid {Colors.BORDER};
        border-radius: {BorderRadius.SM}px;
        padding: {Spacing.SM}px;
    }}
    
    QMenu::item {{
        padding: {Spacing.SM}px {Spacing.MD}px;
        border-radius: {BorderRadius.SM}px;
    }}
    
    QMenu::item:selected {{
        background-color: {Colors.HOVER};
    }}
    
    QMenu::separator {{
        height: 1px;
        background-color: {Colors.BORDER};
        margin: {Spacing.SM}px;
    }}
    
    QSplitter::handle {{
        background-color: {Colors.BORDER};
    }}
    
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    
    QSplitter::handle:vertical {{
        height: 2px;
    }}
    """
    
    app.setStyleSheet(global_style)

# Utility functions for common styling patterns
def create_stat_card_style(accent_color: str = Colors.ACCENT_CYAN) -> str:
    """Create a styled card for statistics display"""
    return f"""
    QWidget {{
        background-color: {Colors.SURFACE};
        border: 1px solid {Colors.BORDER};
        border-left: 4px solid {accent_color};
        border-radius: {BorderRadius.MD}px;
        padding: {Spacing.MD}px;
    }}
    
    QLabel {{
        background-color: transparent;
        border: none;
    }}
    """

def create_alert_style(alert_type: str = "info") -> str:
    """Create styled alert components"""
    color_map = {
        "success": Colors.SUCCESS,
        "warning": Colors.WARNING,
        "error": Colors.ERROR,
        "info": Colors.INFO
    }
    
    color = color_map.get(alert_type, Colors.INFO)
    
    return f"""
    QWidget {{
        background-color: rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1);
        border: 1px solid {color};
        border-radius: {BorderRadius.SM}px;
        padding: {Spacing.MD}px;
    }}
    
    QLabel {{
        color: {color};
        background-color: transparent;
        border: none;
        font-weight: 500;
    }}
    """

class MaterialColors:
    """Alias for Colors for compatibility with old code."""
    primary = Colors.ACCENT_CYAN
    primary_variant = Colors.ACCENT_BLUE
    secondary = Colors.ACCENT_GREEN
    surface = Colors.SURFACE
    text_primary = Colors.TEXT_PRIMARY
    text_on_primary = Colors.TEXT_PRIMARY

class FinaleStyles:
    """Basic style helpers for compatibility with old code."""
    @staticmethod
    def get_group_box_style():
        return """
        QGroupBox {
            border: 1px solid #424242;
            border-radius: 8px;
            margin-top: 8px;
            background-color: #232323;
        }
        QGroupBox:title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
            color: #B0B0B0;
        }
        """
