"""
Advanced Launcher for Smart Intersection Monitoring System
Bhai, yeh advanced main window ke saath sab kuch integrate hai!
"""

from PySide6.QtWidgets import QApplication, QSplashScreen, QLabel
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QFont
import sys
import os
import time
from pathlib import Path

def show_advanced_splash(app):
    """Show advanced splash screen"""
    # Create a simple splash screen
    splash_label = QLabel()
    splash_label.setText("""
    🌉 Smart Intersection Monitoring System
    
    ⚡ Advanced UI Loading...
    
    • 6 Specialized Tabs
    • Real-time Monitoring  
    • AI-Powered Insights
    • IoT Integration
    • Modern Theme System
    """)
    
    splash_label.setAlignment(Qt.AlignCenter)
    splash_label.setStyleSheet("""
        QLabel {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2c3e50, stop:1 #3498db);
            color: white;
            font-size: 14px;
            font-family: 'Segoe UI';
            padding: 40px;
            border-radius: 10px;
        }
    """)
    
    splash_label.setFont(QFont("Segoe UI", 12))
    splash_label.setFixedSize(400, 300)
    
    # Center splash screen
    screen_geometry = app.primaryScreen().geometry()
    splash_geometry = splash_label.frameGeometry()
    center_point = screen_geometry.center()
    splash_geometry.moveCenter(center_point)
    splash_label.move(splash_geometry.topLeft())
    
    splash_label.show()
    app.processEvents()
    
    return splash_label

def main():
    """Advanced main function"""
    print("=" * 60)
    print("🌉 SMART INTERSECTION MONITORING SYSTEM - ADVANCED UI")
    print("=" * 60)
    print(f"🐍 Python: {sys.executable}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print(f"📍 Script Location: {Path(__file__).parent}")
    print("=" * 60)
    
    # Create application
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Smart Intersection - Advanced")
    app.setApplicationVersion("2.0")
    
    # Show splash screen
    print("🎬 Showing splash screen...")
    splash = show_advanced_splash(app)
    
    # Give splash screen time to display
    time.sleep(1.5)
    
    try:
        # Import advanced main window
        print("📦 Loading advanced main window...")
        from advanced_main_window import AdvancedMainWindow
        
        # Create main window
        print("🏗️ Creating advanced main window...")
        window = AdvancedMainWindow()
        
        # Close splash screen
        splash.close()
        
        # Show main window
        print("✨ Showing advanced UI...")
        window.show()
        
        print("🚀 Advanced Smart Intersection Monitoring System Ready!")
        print("=" * 60)
        print("📋 FEATURES LOADED:")
        print("   🎥 Live Monitoring Tab - Multi-camera real-time monitoring")
        print("   📊 Video Analysis Tab - Advanced analytics with ROI")
        print("   🤖 AI Insights Tab - VLM-powered scene analysis") 
        print("   ⚠️ Violations Tab - Traffic violation evidence dashboard")
        print("   📈 Performance Tab - System metrics and monitoring")
        print("   🌉 Smart Intersection - IoT integration and traffic control")
        print("   🎨 Theme Manager - Dark/Light themes with WCAG AAA compliance")
        print("   🔔 Notification Center - Real-time alerts and notifications")
        print("   ⚙️ Settings Dialog - Comprehensive system configuration")
        print("=" * 60)
        print("💡 Tip: Use Ctrl+, for Settings, F11 for Fullscreen")
        print("🌙 Tip: Click theme button in toolbar to switch themes")
        print("=" * 60)
        
        # Start application
        return app.exec()
        
    except ImportError as e:
        splash.close()
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all UI components are properly installed")
        
        # Try fallback
        try:
            print("🔄 Trying fallback UI...")
            from ui.main_window import MainWindow
            window = MainWindow()
            window.show()
            print("✅ Fallback UI loaded")
            return app.exec()
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            return 1
            
    except Exception as e:
        splash.close()
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\n👋 Exiting with code: {exit_code}")
    sys.exit(exit_code)
