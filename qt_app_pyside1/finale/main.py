from PySide6.QtWidgets import QApplication
import sys
import os
import time

def main():
    # Create application instance first
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Show splash screen if available
    splash = None
    try:
        from splash import show_splash
        splash, app = show_splash(app)
    except Exception as e:
        print(f"Could not show splash screen: {e}")

    # Add a short delay to show the splash screen
    if splash:
        time.sleep(1)

    try:
        # Try to use enhanced version with traffic light detection
        from ..ui.main_window import MainWindow
        print("✅ Using standard MainWindow")
    except Exception as e:
        # Fall back to standard version
        print(f"⚠️ Could not load MainWindow: {e}")
        sys.exit(1)

    try:
        # Initialize main window
        window = MainWindow()
        
        # Close splash if it exists
        if splash:
            splash.finish(window)
        
        # Show main window
        window.show()
        
        # Start application event loop
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
