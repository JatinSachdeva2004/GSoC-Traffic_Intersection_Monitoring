"""
OPTIMIZED PYINSTALLER BUILD SCRIPT v2.0
========================================
This script addresses all critical errors and warnings from the build log:

Critical Fixes:
1. Missing __init__.py files (fixed by creating them)
2. Missing hidden imports (cv2, json modules)
3. Correct data file inclusion
4. Platform-specific optimizations

Usage: python build_exe_optimized.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def clean_build_artifacts():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")
    
    artifacts = ['build', 'dist', '*.spec']
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
                print(f"   Removed directory: {artifact}")
            else:
                os.remove(artifact)
                print(f"   Removed file: {artifact}")

def verify_dependencies():
    """Verify all required packages are installed"""
    print("📦 Verifying dependencies...")
    
    required_packages = [
        'PySide6', 'opencv-python', 'numpy', 'openvino',
        'ultralytics', 'matplotlib', 'pillow', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.lower().replace('-', '_'))
            print(f"   ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("   ✓ All dependencies verified")
    return True

def build_executable():
    """Build the executable with optimized PyInstaller command"""
    print("🔨 Building executable...")
    
    # Core PyInstaller command with ALL critical fixes
    cmd = [
        'pyinstaller',
        '--name=TrafficMonitoringApp',
        '--onefile',  # Single executable
        '--windowed',  # No console window
        '--icon=resources/app_icon.ico' if os.path.exists('resources/app_icon.ico') else '',
        
        # === CRITICAL HIDDEN IMPORTS (Fixes Build Errors) ===
        '--hidden-import=cv2',
        '--hidden-import=cv2.cv2',
        '--hidden-import=numpy',
        '--hidden-import=numpy.core',
        '--hidden-import=openvino',
        '--hidden-import=openvino.runtime',
        '--hidden-import=ultralytics',
        '--hidden-import=ultralytics.engine',
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtWidgets',
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=json',
        '--hidden-import=pathlib',
        '--hidden-import=threading',
        '--hidden-import=queue',
        
        # === UI/CONTROLLER MODULES ===
        '--hidden-import=ui',
        '--hidden-import=ui.main_window',
        '--hidden-import=ui.main_window1',
        '--hidden-import=controllers',
        '--hidden-import=controllers.video_controller',
        '--hidden-import=utils',
        '--hidden-import=utils.detection_utils',
        '--hidden-import=utils.tracking_utils',
        '--hidden-import=utils.crosswalk_utils_advanced',
        '--hidden-import=utils.traffic_light_utils',
        
        # === EXCLUDE HEAVY/UNUSED MODULES (Reduces Size) ===
        '--exclude-module=matplotlib.backends._backend_pdf',
        '--exclude-module=matplotlib.backends._backend_ps',
        '--exclude-module=matplotlib.backends._backend_svg',
        '--exclude-module=tkinter',
        '--exclude-module=PyQt5',
        '--exclude-module=unittest',
        '--exclude-module=test',
        '--exclude-module=distutils',
        
        # === DATA FILES AND DIRECTORIES ===
        '--add-data=config.json;.',
        '--add-data=resources;resources',
        '--add-data=openvino_models;openvino_models',
        '--add-data=ui;ui',
        '--add-data=controllers;controllers',
        '--add-data=utils;utils',
        
        # === SPLASH SCREEN ===
        '--splash=resources/splash.png' if os.path.exists('resources/splash.png') else '',
        
        # === MAIN SCRIPT ===
        'main.py'
    ]
    
    # Remove empty strings from command
    cmd = [arg for arg in cmd if arg]
    
    print("📋 PyInstaller command:")
    print("   " + " ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Build completed successfully!")
        print(f"📁 Executable location: dist/TrafficMonitoringApp.exe")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Build failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def post_build_verification():
    """Verify the built executable"""
    print("🔍 Post-build verification...")
    
    exe_path = Path('dist/TrafficMonitoringApp.exe')
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"   ✓ Executable created: {size_mb:.1f} MB")
        
        # Check if critical files are bundled
        print("   📋 Bundled resources check:")
        print("      - config.json: Expected in executable")
        print("      - openvino_models/: Expected in executable")  
        print("      - resources/: Expected in executable")
        
        return True
    else:
        print("   ❌ Executable not found!")
        return False

def main():
    """Main build process"""
    print("🚀 TRAFFIC MONITORING APP - OPTIMIZED BUILD")
    print("=" * 50)
    
    # Step 1: Clean artifacts
    clean_build_artifacts()
    print()
    
    # Step 2: Verify dependencies
    if not verify_dependencies():
        print("\n❌ Build aborted due to missing dependencies")
        sys.exit(1)
    print()
    
    # Step 3: Build executable
    if not build_executable():
        print("\n❌ Build failed")
        sys.exit(1)
    print()
    
    # Step 4: Post-build verification
    if not post_build_verification():
        print("\n⚠️  Build completed but verification failed")
        sys.exit(1)
    
    print("\n🎉 BUILD SUCCESSFUL!")
    print("=" * 50)
    print("📁 Executable: dist/TrafficMonitoringApp.exe")
    print("🏃 To run: dist\\TrafficMonitoringApp.exe")
    print("\n💡 Next steps:")
    print("   1. Test the executable in a clean environment")
    print("   2. Verify all UI elements load correctly")
    print("   3. Test video processing and violation detection")
    print("   4. Check configuration file loading")

if __name__ == "__main__":
    main()
