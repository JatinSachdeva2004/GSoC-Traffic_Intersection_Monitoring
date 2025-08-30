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

    print("🔄 Attempting to load MainWindow1...")
    try:
        # Try to use enhanced version with traffic light detection
        from ui.main_window1 import MainWindow
        print("✅ SUCCESS: Using enhanced MainWindow1 with modern UI")
    except Exception as e:
        print(f"❌ FAILED to load MainWindow1: {e}")
        print("📝 Detailed error:")
        import traceback
        traceback.print_exc()
        
        # Fall back to standard version if main_window1 fails
        print("\n🔄 Attempting fallback to standard MainWindow...")
        try:
            from ui.main_window import MainWindow
            print("⚠️ SUCCESS: Using fallback standard MainWindow")
        except Exception as e2:
            print(f"❌ Could not load MainWindow1: {e}")
            print(f"❌ Fallback MainWindow also failed: {e2}")
            print("\n💡 Please check if these files exist:")
            print("   - ui/main_window1.py")
            print("   - ui/main_window.py")
            print("   - All required UI components")
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
