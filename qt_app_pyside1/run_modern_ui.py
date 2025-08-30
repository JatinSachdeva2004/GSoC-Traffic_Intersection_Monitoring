"""
Run the modern Smart Intersection Monitoring System with the new UI
"""

from PySide6.QtWidgets import QApplication
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main application entry point"""
    # Create application
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Smart Intersection Monitoring System")
    app.setOrganizationName("SmartIntersection")
    app.setOrganizationDomain("smartintersection.com")
    
    try:
        # Import the modern main window
        from ui.modern_main_window import ModernMainWindow
        print("✅ Loading Modern Smart Intersection UI...")
        
        # Create and show main window
        window = ModernMainWindow()
        window.show()
        
        print("🌉 Smart Intersection Monitoring System ready!")
        print("📱 Modern UI with 6 specialized tabs loaded successfully")
        
        # Start the application
        return app.exec()
        
    except ImportError as e:
        print(f"❌ Failed to import modern UI: {e}")
        print("💡 Falling back to standard UI...")
        
        # Fallback to standard UI
        try:
            from ui.main_window import MainWindow
            window = MainWindow()
            window.show()
            return app.exec()
        except Exception as e:
            print(f"❌ Failed to load any UI: {e}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
