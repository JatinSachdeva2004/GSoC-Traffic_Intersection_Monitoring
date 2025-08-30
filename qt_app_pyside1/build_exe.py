#!/usr/bin/env python3
"""
Comprehensive build script for Traffic Monitor application
This script handles the complete build process with all necessary PyInstaller flags
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def build_application():
    """Build the application with PyInstaller"""
    
    # Get current directory
    current_dir = Path.cwd()
    print(f"Building from: {current_dir}")
    
    # Clean previous builds
    print("\n🧹 Cleaning previous builds...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed {folder}")
    
    if os.path.exists('TrafficMonitor.spec'):
        os.remove('TrafficMonitor.spec')
        print("Removed old spec file")
    
    # Define PyInstaller command with all necessary flags
    pyinstaller_cmd = [
        'pyinstaller',
        '--name=TrafficMonitor',
        '--windowed',  # Remove for debugging
        '--onefile',
        '--icon=resources/icon.ico' if os.path.exists('resources/icon.ico') else '',
        
        # Add data files and folders
        '--add-data=ui;ui',
        '--add-data=controllers;controllers', 
        '--add-data=utils;utils',
        '--add-data=openvino_models;openvino_models',
        '--add-data=resources;resources' if os.path.exists('resources') else '',
        '--add-data=config.json;.',
        '--add-data=splash.py;.',
        
        # Hidden imports for modules PyInstaller might miss
        '--hidden-import=cv2',
        '--hidden-import=openvino',
        '--hidden-import=numpy',
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtWidgets', 
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=json',
        '--hidden-import=os',
        '--hidden-import=sys',
        '--hidden-import=time',
        '--hidden-import=traceback',
        '--hidden-import=pathlib',
        
        # Main script
        'main.py'
    ]
    
    # Remove empty icon parameter if no icon exists
    pyinstaller_cmd = [arg for arg in pyinstaller_cmd if arg and not arg.startswith('--icon=') or os.path.exists(arg.split('=')[1] if '=' in arg else '')]
    
    # Convert to string command
    cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in pyinstaller_cmd)
    
    # Build the application
    if run_command(cmd_str, "Building Traffic Monitor application"):
        print(f"\n✅ Build completed successfully!")
        print(f"Executable location: {current_dir}/dist/TrafficMonitor.exe")
        return True
    else:
        print(f"\n❌ Build failed!")
        return False

def build_debug_version():
    """Build a debug version with console output"""
    
    print("\n🔧 Building debug version...")
    
    # Define PyInstaller command for debug build
    pyinstaller_cmd = [
        'pyinstaller',
        '--name=TrafficMonitorDebug',
        '--console',  # Enable console for debugging
        '--onefile',
        
        # Add data files and folders
        '--add-data=ui;ui',
        '--add-data=controllers;controllers', 
        '--add-data=utils;utils',
        '--add-data=openvino_models;openvino_models',
        '--add-data=resources;resources' if os.path.exists('resources') else '',
        '--add-data=config.json;.',
        '--add-data=splash.py;.',
        
        # Hidden imports
        '--hidden-import=cv2',
        '--hidden-import=openvino',
        '--hidden-import=numpy',
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtWidgets', 
        '--hidden-import=PySide6.QtGui',
        '--hidden-import=json',
        '--hidden-import=os',
        '--hidden-import=sys',
        '--hidden-import=time',
        '--hidden-import=traceback',
        '--hidden-import=pathlib',
        
        # Main script
        'main.py'
    ]
    
    # Convert to string command
    cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in pyinstaller_cmd)
    
    return run_command(cmd_str, "Building debug version")

def main():
    """Main build process"""
    print("🚀 Traffic Monitor Build Script")
    print("=" * 50)
    
    # Check if PyInstaller is available
    try:
        subprocess.run(['pyinstaller', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyInstaller not found. Installing...")
        if not run_command('pip install pyinstaller', "Installing PyInstaller"):
            print("Failed to install PyInstaller")
            return False
    
    # Check for required files
    required_files = ['main.py', 'ui', 'controllers', 'utils', 'config.json']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files/folders: {missing_files}")
        return False
    
    print("✅ All required files found")
    
    # Build debug version first
    if build_debug_version():
        print("\n✅ Debug build completed!")
        print(f"Debug executable: {Path.cwd()}/dist/TrafficMonitorDebug.exe")
    
    # Build main application
    if build_application():
        print(f"\n🎉 All builds completed successfully!")
        print(f"Main executable: {Path.cwd()}/dist/TrafficMonitor.exe")
        print(f"Debug executable: {Path.cwd()}/dist/TrafficMonitorDebug.exe")
        
        print(f"\n📝 To test:")
        print(f"1. Run debug version first: dist\\TrafficMonitorDebug.exe")
        print(f"2. If working, run main version: dist\\TrafficMonitor.exe")
        
        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
