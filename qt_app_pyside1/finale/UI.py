"""
Finale UI - Main Entry Point
Modern traffic monitoring interface entry point.
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor
import sys
import os
from pathlib import Path

# Import finale components
try:
    # Try relative imports first (when running as a package)
    from .main_window import FinaleMainWindow
    from .splash import FinaleSplashScreen
    from .styles import FinaleStyles, MaterialColors
    from .icons import FinaleIcons
except ImportError:
    # Fallback to direct imports (when running as script)
    try:
        from main_window import FinaleMainWindow
        from splash import FinaleSplashScreen
        from styles import FinaleStyles, MaterialColors
        from icons import FinaleIcons
    except ImportError:
        print('Error importing main components')

# Add Qt message handler from original main.py
def qt_message_handler(mode, context, message):
    print(f"Qt Message: {message} (Mode: {mode})")
# Install custom handler for Qt messages
from PySide6.QtCore import Qt
if hasattr(Qt, 'qInstallMessageHandler'):
    Qt.qInstallMessageHandler(qt_message_handler)

class FinaleUI:
    """
    Main Finale UI application class.
    Handles application initialization, theme setup, and window management.
    """
    
    def __init__(self):
        self.app = None
        self.main_window = None
        self.splash = None
        
    def initialize_application(self, sys_argv=None):
        """
        Initialize the QApplication with proper settings.
        
        Args:
            sys_argv: System arguments (defaults to sys.argv)
        """
        if sys_argv is None:
            sys_argv = sys.argv
            
        # Create or get existing application instance
        self.app = QApplication.instance() or QApplication(sys_argv)
        
        # Set application properties
        self.app.setApplicationName("Finale Traffic Monitoring")
        self.app.setApplicationVersion("1.0.0")
        self.app.setOrganizationName("Finale Systems")
        self.app.setOrganizationDomain("finale.traffic")
        
        # Set application icon
        self.app.setWindowIcon(FinaleIcons.get_icon("traffic_monitoring"))
        
        # Enable high DPI scaling
        self.app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        self.app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Set font
        self.setup_fonts()
        
        # Set global theme
        self.setup_theme()
        
        return self.app
    
    def setup_fonts(self):
        """Setup application fonts"""
        # Set default font
        font = QFont("Segoe UI", 9)
        font.setHintingPreference(QFont.PreferDefaultHinting)
        self.app.setFont(font)
    
    def setup_theme(self):
        """Setup global application theme"""
        # Apply dark theme by default
        MaterialColors.apply_dark_theme()
        
        # Set global stylesheet
        self.app.setStyleSheet(FinaleStyles.get_global_style())
    
    def show_splash_screen(self):
        """Show splash screen during initialization"""
        try:
            self.splash = FinaleSplashScreen()
            self.splash.show()
            
            # Process events to show splash
            self.app.processEvents()
            
            return self.splash
        except Exception as e:
            print(f"Could not show splash screen: {e}")
            return None
    
    def create_main_window(self):
        """Create and initialize the main window"""
        try:
            self.main_window = FinaleMainWindow()
            return self.main_window
        except Exception as e:
            print(f"Error creating main window: {e}")
            raise
    
    def run(self, show_splash=True):
        """
        Run the complete Finale UI application.
        
        Args:
            show_splash: Whether to show splash screen
            
        Returns:
            Application exit code
        """
        try:
            # Initialize application
            if not self.app:
                self.initialize_application()
            
            # Show splash screen
            if show_splash:
                splash = self.show_splash_screen()
                if splash:
                    splash.update_progress(20, "Initializing UI components...")
                    self.app.processEvents()
            
            # Create main window
            if splash:
                splash.update_progress(50, "Loading detection models...")
                self.app.processEvents()
            
            self.main_window = self.create_main_window()
            
            if splash:
                splash.update_progress(80, "Connecting to backend...")
                self.app.processEvents()
            
            # Finish splash and show main window
            if splash:
                splash.update_progress(100, "Ready!")
                self.app.processEvents()
                splash.finish(self.main_window)
            
            # Show main window
            self.main_window.show()
            
            # Start event loop
            return self.app.exec()
            
        except Exception as e:
            print(f"❌ Error running Finale UI: {e}")
            import traceback
            traceback.print_exc()
            return 1

def create_finale_app(sys_argv=None):
    """
    Create and return a Finale UI application instance.
    
    Args:
        sys_argv: System arguments
        
    Returns:
        FinaleUI instance
    """
    finale_ui = FinaleUI()
    finale_ui.initialize_application(sys_argv)
    return finale_ui

def run_finale_ui(sys_argv=None, show_splash=True):
    """
    Convenience function to run the Finale UI.
    
    Args:
        sys_argv: System arguments
        show_splash: Whether to show splash screen
        
    Returns:
        Application exit code
    """
    finale_ui = create_finale_app(sys_argv)
    return finale_ui.run(show_splash)

# Main execution
if __name__ == "__main__":
    exit_code = run_finale_ui()
    sys.exit(exit_code)
