"""
Icon Management System
=====================

Comprehensive icon system with SVG icons, Material Design icons,
and utility functions for the Traffic Monitoring Application.

Features:
- Material Design icon set
- SVG icon generation
- Icon theming and colorization
- Size variants and scaling
- Custom icon registration
"""

from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush, QPen
from PySide6.QtCore import Qt, QSize
from PySide6.QtSvg import QSvgRenderer
from typing import Dict, Optional, Tuple
import base64
from io import BytesIO

class IconTheme:
    """Icon theme management"""
    
    # Icon colors for dark theme
    PRIMARY = "#FFFFFF"
    SECONDARY = "#B0B0B0"
    ACCENT = "#00BCD4"
    SUCCESS = "#4CAF50"
    WARNING = "#FF9800"
    ERROR = "#F44336"
    INFO = "#2196F3"

class SVGIcons:
    """Collection of SVG icons as base64 encoded strings"""
    
    # Navigation icons
    HOME = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2l7 7v11h-4v-7h-6v7H5V9l7-7z"/>
    </svg>
    """
    
    PLAY = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M8 5v14l11-7z"/>
    </svg>
    """
    
    PAUSE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
    </svg>
    """
    
    STOP = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 6h12v12H6z"/>
    </svg>
    """
    
    RECORD = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="8"/>
    </svg>
    """
    
    # Detection and monitoring icons
    CAMERA = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 8.8c-2.1 0-3.8 1.7-3.8 3.8s1.7 3.8 3.8 3.8 3.8-1.7 3.8-3.8-1.7-3.8-3.8-3.8z"/>
        <path d="M21 7h-3.4l-1.9-2.6c-.4-.5-.9-.8-1.6-.8H9.9c-.7 0-1.2.3-1.6.8L6.4 7H3c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V9c0-1.1-.9-2-2-2z"/>
    </svg>
    """
    
    MONITOR = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M21 3H3c-1.1 0-2 .9-2 2v11c0 1.1.9 2 2 2h6l-2 3v1h8v-1l-2-3h6c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 11H3V5h18v9z"/>
    </svg>
    """
    
    TRAFFIC_LIGHT = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <rect x="8" y="2" width="8" height="20" rx="4" stroke="currentColor" stroke-width="2" fill="none"/>
        <circle cx="12" cy="7" r="2" fill="#F44336"/>
        <circle cx="12" cy="12" r="2" fill="#FF9800"/>
        <circle cx="12" cy="17" r="2" fill="#4CAF50"/>
    </svg>
    """
    
    VIOLATION = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2L1 21h22L12 2zm0 3.99L19.53 19H4.47L12 5.99zM11 16h2v2h-2v-2zm0-6h2v4h-2v-4z"/>
    </svg>
    """
    
    # Analytics and statistics icons
    CHART_BAR = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M5 9v6h4V9H5zm6-4v10h4V5h-4zm6 6v4h4v-4h-4z"/>
    </svg>
    """
    
    CHART_LINE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M3.5 18.49l6-6.01 4 4L22 6.92l-1.41-1.41-7.09 7.97-4-4L3 16.99l.5 1.5z"/>
    </svg>
    """
    
    CHART_PIE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M11 2v20c-5.07-.5-9-4.79-9-10s3.93-9.5 9-10zm2.03 0v8.99H22c-.47-4.74-4.24-8.52-8.97-8.99zm0 11.01V22c4.74-.47 8.5-4.25 8.97-8.99h-8.97z"/>
    </svg>
    """
    
    DASHBOARD = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
    </svg>
    """
    
    # System and settings icons
    SETTINGS = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
    </svg>
    """
    
    EXPORT = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2L8 8h3v8h2V8h3l-4-6zm7 7h-2v10H7V9H5v10c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2V9z"/>
    </svg>
    """
    
    IMPORT = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 22L8 16h3V8h2v8h3l-4 6zm7-15h-2V5H7v2H5V5c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2v2z"/>
    </svg>
    """
    
    SAVE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V6h10v3z"/>
    </svg>
    """
    
    # Status and alert icons
    CHECK_CIRCLE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
    </svg>
    """
    
    WARNING_CIRCLE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
    </svg>
    """
    
    ERROR_CIRCLE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z"/>
    </svg>
    """
    
    INFO_CIRCLE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
    </svg>
    """
    
    # Action icons
    REFRESH = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
    </svg>
    """
    
    DELETE = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
    </svg>
    """
    
    EDIT = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z"/>
    </svg>
    """
    
    FILTER = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/>
    </svg>
    """
    
    SEARCH = """
    <svg viewBox="0 0 24 24" fill="currentColor">
        <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
    </svg>
    """

class IconManager:
    """Manages icons for the application"""
    
    def __init__(self):
        self._icon_cache: Dict[str, QIcon] = {}
        self.theme = IconTheme()
    
    def get_icon(self, name: str, color: str = IconTheme.PRIMARY, size: int = 24) -> QIcon:
        """Get an icon by name with specified color and size"""
        cache_key = f"{name}_{color}_{size}"
        
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]
        
        # Get SVG content
        svg_content = getattr(SVGIcons, name.upper(), None)
        if not svg_content:
            return QIcon()  # Return empty icon if not found
        
        # Replace currentColor with specified color
        svg_content = svg_content.replace('currentColor', color)
        
        # Create icon from SVG
        icon = self._create_icon_from_svg(svg_content, size)
        self._icon_cache[cache_key] = icon
        
        return icon
    
    def _create_icon_from_svg(self, svg_content: str, size: int) -> QIcon:
        """Create QIcon from SVG content"""
        # Create QSvgRenderer from SVG content
        svg_bytes = svg_content.encode('utf-8')
        renderer = QSvgRenderer(svg_bytes)
        
        # Create pixmap
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        # Paint SVG onto pixmap
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        
        return QIcon(pixmap)
    
    def get_status_icon(self, status: str, size: int = 16) -> QIcon:
        """Get icon for specific status"""
        status_map = {
            'success': ('CHECK_CIRCLE', IconTheme.SUCCESS),
            'warning': ('WARNING_CIRCLE', IconTheme.WARNING),
            'error': ('ERROR_CIRCLE', IconTheme.ERROR),
            'info': ('INFO_CIRCLE', IconTheme.INFO),
            'violation': ('VIOLATION', IconTheme.ERROR),
            'active': ('PLAY', IconTheme.SUCCESS),
            'inactive': ('PAUSE', IconTheme.SECONDARY),
            'recording': ('RECORD', IconTheme.ERROR)
        }
        
        icon_name, color = status_map.get(status, ('INFO_CIRCLE', IconTheme.INFO))
        return self.get_icon(icon_name, color, size)
    
    def get_action_icon(self, action: str, size: int = 20) -> QIcon:
        """Get icon for specific action"""
        action_map = {
            'play': 'PLAY',
            'pause': 'PAUSE',
            'stop': 'STOP',
            'record': 'RECORD',
            'settings': 'SETTINGS',
            'export': 'EXPORT',
            'import': 'IMPORT',
            'save': 'SAVE',
            'refresh': 'REFRESH',
            'delete': 'DELETE',
            'edit': 'EDIT',
            'filter': 'FILTER',
            'search': 'SEARCH'
        }
        
        icon_name = action_map.get(action, 'INFO_CIRCLE')
        return self.get_icon(icon_name, IconTheme.PRIMARY, size)
    
    def get_navigation_icon(self, view: str, size: int = 24) -> QIcon:
        """Get icon for navigation views"""
        nav_map = {
            'home': 'HOME',
            'detection': 'CAMERA',
            'violations': 'VIOLATION',
            'analytics': 'DASHBOARD',
            'export': 'EXPORT',
            'monitor': 'MONITOR',
            'chart': 'CHART_BAR'
        }
        
        icon_name = nav_map.get(view, 'HOME')
        return self.get_icon(icon_name, IconTheme.ACCENT, size)
    
    def create_colored_icon(self, base_icon: str, color: str, size: int = 24) -> QIcon:
        """Create a colored version of an icon"""
        return self.get_icon(base_icon, color, size)
    
    def set_theme_color(self, color: str):
        """Set the theme accent color"""
        self.theme.ACCENT = color
        # Clear cache to regenerate icons with new color
        self._icon_cache.clear()

# Global icon manager instance
icon_manager = IconManager()

# Convenience functions
def get_icon(name: str, color: str = IconTheme.PRIMARY, size: int = 24) -> QIcon:
    """Get an icon - convenience function"""
    return icon_manager.get_icon(name, color, size)

def get_status_icon(status: str, size: int = 16) -> QIcon:
    """Get status icon - convenience function"""
    return icon_manager.get_status_icon(status, size)

def get_action_icon(action: str, size: int = 20) -> QIcon:
    """Get action icon - convenience function"""
    return icon_manager.get_action_icon(action, size)

def get_navigation_icon(view: str, size: int = 24) -> QIcon:
    """Get navigation icon - convenience function"""
    return icon_manager.get_navigation_icon(view, size)

# Common icon sets for easy access
class CommonIcons:
    """Commonly used icon combinations"""
    
    @staticmethod
    def toolbar_icons() -> Dict[str, QIcon]:
        """Get all toolbar icons"""
        return {
            'play': get_action_icon('play'),
            'pause': get_action_icon('pause'),
            'stop': get_action_icon('stop'),
            'record': get_action_icon('record'),
            'settings': get_action_icon('settings'),
            'export': get_action_icon('export'),
            'refresh': get_action_icon('refresh')
        }
    
    @staticmethod
    def status_icons() -> Dict[str, QIcon]:
        """Get all status icons"""
        return {
            'success': get_status_icon('success'),
            'warning': get_status_icon('warning'),
            'error': get_status_icon('error'),
            'info': get_status_icon('info'),
            'violation': get_status_icon('violation'),
            'active': get_status_icon('active'),
            'inactive': get_status_icon('inactive'),
            'recording': get_status_icon('recording')
        }
    
    @staticmethod
    def navigation_icons() -> Dict[str, QIcon]:
        """Get all navigation icons"""
        return {
            'detection': get_navigation_icon('detection'),
            'violations': get_navigation_icon('violations'),
            'analytics': get_navigation_icon('analytics'),
            'export': get_navigation_icon('export'),
            'monitor': get_navigation_icon('monitor')
        }

# Traffic light specific icons
def create_traffic_light_icon(red_on: bool = False, yellow_on: bool = False, green_on: bool = False, size: int = 32) -> QIcon:
    """Create a traffic light icon with specific lights on/off"""
    svg_template = f"""
    <svg viewBox="0 0 24 24" width="{size}" height="{size}">
        <rect x="8" y="2" width="8" height="20" rx="4" stroke="#424242" stroke-width="2" fill="#2C2C2C"/>
        <circle cx="12" cy="7" r="2" fill="{'#F44336' if red_on else '#5D4037'}"/>
        <circle cx="12" cy="12" r="2" fill="{'#FF9800' if yellow_on else '#5D4037'}"/>
        <circle cx="12" cy="17" r="2" fill="{'#4CAF50' if green_on else '#5D4037'}"/>
    </svg>
    """
    
    svg_bytes = svg_template.encode('utf-8')
    renderer = QSvgRenderer(svg_bytes)
    
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    
    return QIcon(pixmap)

# New FinaleIcons class to wrap the existing functionality
class FinaleIcons:
    """
    Wrapper class for icon management to maintain compatibility
    with existing code that references FinaleIcons.get_icon() etc.
    """
    
    @staticmethod
    def get_icon(name: str, color: str = IconTheme.PRIMARY, size: int = 24) -> QIcon:
        """Get an icon by name"""
        return get_icon(name, color, size)
    
    @staticmethod
    def get_status_icon(status: str, size: int = 16) -> QIcon:
        """Get a status icon"""
        return get_status_icon(status, size)
    
    @staticmethod
    def get_action_icon(action: str, size: int = 20) -> QIcon:
        """Get an action icon"""
        return get_action_icon(action, size)
    
    @staticmethod
    def get_navigation_icon(view: str, size: int = 24) -> QIcon:
        """Get a navigation icon"""
        return get_navigation_icon(view, size)
    
    @staticmethod
    def create_colored_icon(base_icon: str, color: str, size: int = 24) -> QIcon:
        """Create a colored version of an icon"""
        return get_icon(base_icon, color, size)
    
    @staticmethod
    def traffic_light_icon(red_on: bool = False, yellow_on: bool = False, green_on: bool = False, size: int = 32) -> QIcon:
        """Create a traffic light icon with specific lights on/off"""
        return create_traffic_light_icon(red_on, yellow_on, green_on, size)
