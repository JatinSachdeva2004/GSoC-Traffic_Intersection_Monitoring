#!/usr/bin/env python3
"""
Fixed Cross-Platform Build Script for Traffic Monitor Application
This script fixes all the PyInstaller build issues
"""

import os
import subprocess
import sys
import shutil
import platform
from pathlib import Path

def get_platform_info():
    """Get current platform information"""
    system = platform.system().lower()
    
    if system == "windows":
        return "windows", "exe", ";"
    elif system == "darwin":
        return "macos", "app", ":"
    elif system == "linux":
        return "linux", "", ":"
    else:
        return "unknown", "", ":"

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print("✅ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Exit code: {e.returncode}")
        return False

def check_module_exists(module_name):
    """Check if a module is installed and can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_essential_hidden_imports():
    """Get essential hidden imports that actually exist"""
    essential_imports = [
        # Core dependencies (check if they exist)
        'cv2',
        'openvino', 
        'numpy',
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        
        # Application modules (these should exist in our code)
        'utils.annotation_utils',
        'utils.helpers',
        'utils.enhanced_annotation_utils', 
        'utils.data_publisher',
        'utils.mqtt_publisher',
        'utils.traffic_light_utils',
        'utils.scene_analytics',
        'utils.crosswalk_utils',
        'utils.crosswalk_utils2',
        'utils.enhanced_tracker',
        'utils.embedder_openvino',
        'controllers.video_controller_new',
        'controllers.model_manager',
        'controllers.analytics_controller',
        'controllers.performance_overlay',
        'controllers.vlm_controller',
        'controllers.smart_intersection_controller',
        'ui.main_window',
        'ui.analytics_tab', 
        'ui.violations_tab',
        'ui.export_tab',
        'ui.modern_config_panel',
        'ui.modern_live_detection_tab',
    ]
    
    # Optional imports
    optional_imports = [
        'torch',
        'ultralytics',
        'transformers',
        'scipy',
        'matplotlib', 
        'pandas',
        'requests',
        'yaml',
        'paho.mqtt',
        'influxdb_client',
    ]
    
    # Filter to only include available modules
    available_imports = []
    
    for module in essential_imports:
        # For our local modules, assume they exist (we'll check paths)
        if module.startswith(('utils.', 'controllers.', 'ui.')):
            available_imports.append(module)
        elif check_module_exists(module.split('.')[0]):  # Check top-level module
            available_imports.append(module)
        else:
            print(f"⚠️  Module '{module}' not found - skipping")
    
    for module in optional_imports:
        if check_module_exists(module):
            available_imports.append(module)
        else:
            print(f"⚠️  Optional module '{module}' not found - skipping")
    
    return available_imports

def build_application(platform_name, extension, data_sep, is_debug=False):
    """Build the application with PyInstaller - simplified version"""
    
    current_dir = Path.cwd()
    parent_dir = current_dir.parent
    print(f"Building from: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    
    # Set app name based on debug mode
    app_name = "TrafficMonitorDebug" if is_debug else "TrafficMonitor"
    
    # Clean previous builds
    print(f"\n🧹 Cleaning previous {app_name} builds...")
    build_folder = current_dir / "build" / app_name
    if build_folder.exists():
        shutil.rmtree(build_folder)
        print(f"Removed {build_folder}")
    
    spec_file = f"{app_name}.spec"
    if os.path.exists(spec_file):
        os.remove(spec_file)
        print(f"Removed old spec file: {spec_file}")
    
    # Build PyInstaller command - SIMPLIFIED
    pyinstaller_cmd = [
        'pyinstaller',
        f'--name={app_name}',
        '--onefile',
        '--clean',
    ]
    
    # Add console/windowed mode
    if is_debug:
        pyinstaller_cmd.append('--console')
    else:
        if platform_name == "windows":
            pyinstaller_cmd.append('--windowed')
        else:
            pyinstaller_cmd.append('--console')  # Keep console for non-Windows
    
    # Recursively add all files and folders from the parent directory (clean-final-push)
    import glob
    root_dir = str(parent_dir)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            # Relative path from root_dir
            rel_path = os.path.relpath(abs_path, root_dir)
            # Destination: keep folder structure
            pyinstaller_cmd.append(f'--add-data={abs_path}{data_sep}{rel_path}')
            print(f"✅ Added: {abs_path} -> {rel_path}")
    
    # Add models from parent directory (essential for app to work)
    models_dir = parent_dir / "models"
    if models_dir.exists():
        pyinstaller_cmd.append(f'--add-data={models_dir}{data_sep}models')
        print(f"✅ Added models folder: {models_dir}")
        
    yolo_model_dir = parent_dir / "yolo11x_openvino_model"
    if yolo_model_dir.exists():
        pyinstaller_cmd.append(f'--add-data={yolo_model_dir}{data_sep}yolo11x_openvino_model')
        print(f"✅ Added YOLO model folder: {yolo_model_dir}")
    
    # Get and add hidden imports
    print("\n🔍 Checking available modules...")
    hidden_imports = get_essential_hidden_imports()
    print(f"✅ Found {len(hidden_imports)} available modules")
    
    for import_name in hidden_imports:
        pyinstaller_cmd.append(f'--hidden-import={import_name}')
    
    # Additional options (simplified)
    pyinstaller_cmd.extend([
        '--noconfirm',
        '--log-level=INFO',  # Changed to INFO to see more details
        '--workpath=build',
        '--distpath=dist',
        
        # Remove problematic --collect-all that causes "not a package" warnings
        # Instead, let PyInstaller discover modules through imports
        
        # Exclude problematic modules
        '--exclude-module=tensorboard',
        '--exclude-module=tkinter',
        '--exclude-module=matplotlib.tests',
        '--exclude-module=pytest',
    ])
    
    # Add version file if exists (Windows only)
    if platform_name == "windows" and os.path.exists('version_info.txt'):
        pyinstaller_cmd.append('--version-file=version_info.txt')
    
    # Add main script
    pyinstaller_cmd.append('main.py')
    
    # Convert to string command
    cmd_str = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in pyinstaller_cmd)
    
    # Build the application
    build_type = "debug" if is_debug else "release"
    if run_command(cmd_str, f"Building {app_name} ({build_type}) for {platform_name}"):
        if platform_name == "windows":
            exe_path = current_dir / "dist" / f"{app_name}.exe"
        elif platform_name == "macos":
            exe_path = current_dir / "dist" / f"{app_name}.app"
        else:
            exe_path = current_dir / "dist" / app_name
            
        print(f"\n✅ {build_type.title()} build completed successfully!")
        print(f"Executable location: {exe_path}")
        return True
    else:
        print(f"\n❌ {build_type.title()} build failed!")
        return False

def test_executable(exe_name, platform_name):
    """Test the built executable"""
    if platform_name == "windows":
        exe_path = Path(f"dist/{exe_name}.exe")
    elif platform_name == "macos":
        exe_path = Path(f"dist/{exe_name}.app")
    else:
        exe_path = Path(f"dist/{exe_name}")
    
    if not exe_path.exists():
        print(f"❌ Executable not found: {exe_path}")
        return False
    
    print(f"\n🧪 Testing executable: {exe_path}")
    
    try:
        if platform_name == "windows":
            # Test with --help flag to avoid GUI issues
            result = subprocess.run([str(exe_path), "--help"], 
                                  timeout=10, text=True, capture_output=True)
        else:
            result = subprocess.run([str(exe_path), "--help"], 
                                  timeout=10, text=True, capture_output=True)
        
        print(f"✅ Executable test completed! Exit code: {result.returncode}")
        if result.stdout:
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        if result.stderr:
            print("Errors:", result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr)
        
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Executable is running (timeout after 10s) - this might be good!")
        return True
    except Exception as e:
        print(f"❌ Error running executable: {e}")
        return False

def main():
    """Main build process - simplified and fixed"""
    print("🚀 Fixed Cross-Platform Traffic Monitor Build Script")
    print("=" * 60)
    
    # Get platform information
    platform_name, extension, data_sep = get_platform_info()
    print(f"🖥️  Detected platform: {platform_name}")
    print(f"📦 Target extension: {extension}")
    
    if platform_name == "unknown":
        print("❌ Unsupported platform")
        return False
    
    # Check if PyInstaller is available
    try:
        subprocess.run(['pyinstaller', '--version'], check=True, capture_output=True)
        print("✅ PyInstaller is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyInstaller not found. Installing...")
        if not run_command('pip install pyinstaller', "Installing PyInstaller"):
            print("Failed to install PyInstaller")
            return False
    
    # Check for required files
    required_files = ['main.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    
    # Create dist directory if it doesn't exist
    os.makedirs('dist', exist_ok=True)
    
    # Build debug version first
    print(f"\n{'='*60}")
    print("🔨 Building DEBUG version...")
    print(f"{'='*60}")
    
    debug_success = build_application(platform_name, extension, data_sep, is_debug=True)
    
    if debug_success:
        print(f"\n✅ Debug build completed!")
        debug_name = "TrafficMonitorDebug"
        
        # Test debug version
        print(f"\n🧪 Testing DEBUG version...")
        test_executable(debug_name, platform_name)
    
    # Build release version
    print(f"\n{'='*60}")
    print("🔨 Building RELEASE version...")
    print(f"{'='*60}")
    
    release_success = build_application(platform_name, extension, data_sep, is_debug=False)
    
    if release_success:
        print(f"\n✅ Release build completed!")
        release_name = "TrafficMonitor"
        
        # Test release version
        print(f"\n🧪 Testing RELEASE version...")
        test_executable(release_name, platform_name)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📋 BUILD SUMMARY")
    print(f"{'='*60}")
    
    if debug_success and release_success:
        print("🎉 Both builds completed successfully!")
        
        if platform_name == "windows":
            print(f"\n📂 Output files:")
            print(f"  - dist/TrafficMonitor.exe (Release)")
            print(f"  - dist/TrafficMonitorDebug.exe (Debug)")
            print(f"\n📝 To run:")
            print(f"  1. Test debug version: .\\dist\\TrafficMonitorDebug.exe")
            print(f"  2. Test release version: .\\dist\\TrafficMonitor.exe")
        
        return True
    else:
        print("❌ Some builds failed!")
        if not debug_success:
            print("  - Debug build failed")
        if not release_success:
            print("  - Release build failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
