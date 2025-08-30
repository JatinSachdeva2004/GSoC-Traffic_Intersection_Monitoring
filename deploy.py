"""
Deployment script for packaging the Qt app as a standalone executable
"""

import os
import sys
import shutil
import platform
from pathlib import Path

# Get the current directory (where this script is)
CURRENT_DIR = Path(__file__).parent.absolute()
APP_DIR = CURRENT_DIR / "qt_app_pyside"

# Determine platform-specific details
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == "Windows"
IS_LINUX = PLATFORM == "Linux"
IS_MACOS = PLATFORM == "Darwin"

# Path separator for PyInstaller
PATH_SEP = ";" if IS_WINDOWS else ":"

def find_resource_files():
    """Find UI, QRC, and other resource files"""
    resources = []
    
    # Process UI files
    ui_files = list(APP_DIR.glob("**/*.ui"))
    for ui_file in ui_files:
        rel_path = ui_file.relative_to(CURRENT_DIR)
        print(f"Found UI file: {rel_path}")
        # Convert UI files to Python
        output_path = ui_file.with_suffix(".py")
        convert_ui_cmd = f"pyside6-uic {ui_file} -o {output_path}"
        print(f"Converting UI: {convert_ui_cmd}")
        os.system(convert_ui_cmd)
    
    # Process QRC files (resource files)
    qrc_files = list(APP_DIR.glob("**/*.qrc"))
    for qrc_file in qrc_files:
        rel_path = qrc_file.relative_to(CURRENT_DIR)
        print(f"Found QRC file: {rel_path}")
        # Convert QRC files to Python
        output_path = qrc_file.with_suffix("_rc.py")
        convert_qrc_cmd = f"pyside6-rcc {qrc_file} -o {output_path}"
        print(f"Converting QRC: {convert_qrc_cmd}")
        os.system(convert_qrc_cmd)
    
    # Find asset directories
    asset_dirs = [
        "assets", 
        "resources", 
        "images", 
        "icons", 
        "themes",
        "models"
    ]
    
    data_files = []
    for asset_dir in asset_dirs:
        full_path = APP_DIR / asset_dir
        if full_path.exists() and full_path.is_dir():
            rel_path = full_path.relative_to(CURRENT_DIR)
            data_files.append(f"{rel_path}{PATH_SEP}{rel_path}")
            print(f"Found asset directory: {rel_path}")
    
    # Include specific model directories from root if they exist
    root_model_dirs = [
        "models/yolo11x_openvino_model",
        "openvino_models",
        "yolo11x_openvino_model"
    ]
    
    for model_dir in root_model_dirs:
        model_path = Path(CURRENT_DIR) / model_dir
        if model_path.exists() and model_path.is_dir():
            data_files.append(f"{model_dir}{PATH_SEP}{model_dir}")
            print(f"Found model directory: {model_dir}")
    
    # Find specific asset files
    asset_extensions = [".png", ".ico", ".jpg", ".svg", ".json", ".xml", ".bin", ".qss"]
    for ext in asset_extensions:
        for asset_file in APP_DIR.glob(f"**/*{ext}"):
            # Skip files in asset directories we've already included
            if any(dir_name in str(asset_file) for dir_name in asset_dirs):
                continue
                
            # Include individual file
            rel_path = asset_file.relative_to(CURRENT_DIR)
            dir_path = rel_path.parent
            data_files.append(f"{rel_path}{PATH_SEP}{dir_path}")
            print(f"Found asset file: {rel_path}")
    
    return data_files

def create_spec_file(data_files, main_script="main.py"):
    """Create a PyInstaller spec file"""
    spec_path = CURRENT_DIR / "qt_app.spec"    # Format data_files for the spec file
    formatted_data_files = []
    for data_file in data_files:
        src, dst = data_file.split(PATH_SEP)
        # Ensure correct escaping for Windows paths
        if IS_WINDOWS:
            src = src.replace('\\', '\\\\')
            dst = dst.replace('\\', '\\\\')
        formatted_data_files.append(f"(r'{src}', r'{dst}')")
    
    data_files_str = ", ".join(formatted_data_files)
      # Main script location
    main_script_path = APP_DIR / main_script
    if not main_script_path.exists():
        print(f"ERROR: Main script not found at {main_script_path}")
        sys.exit(1)
        
    # Convert path to string with proper escaping
    main_script_path_str = str(main_script_path)
      # Icon file
    icon_file = str(APP_DIR / "resources" / "icon.ico") if IS_WINDOWS else str(APP_DIR / "resources" / "icon.icns")
    if not Path(icon_file).exists():
        icon_file = None
        print("No icon file found. Continuing without an icon.")
    
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    [r'{main_script_path_str}'],
    pathex=['{CURRENT_DIR}'],
    binaries=[],
    datas=[{data_files_str}],
    hiddenimports=['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets'],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],    name='traffic_monitoring_app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
"""
    
    # Add icon if it exists
    if icon_file:
        spec_content += f"    icon=r'{icon_file}',\n"
    
    spec_content += ")\n\n"
    
    # For macOS, create app bundle
    if IS_MACOS:
        spec_content += f"""app = BUNDLE(exe,
             name="TrafficMonitoring.app",
             icon={icon_file},
)
"""
    
    with open(spec_path, "w") as f:
        f.write(spec_content)
    
    print(f"Created PyInstaller spec file: {spec_path}")
    return spec_path

def create_splash_screen_script():
    """Create a splash screen script"""
    splash_script = APP_DIR / "splash.py"
    
    content = """from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
import sys
import os

def show_splash():
    app = QApplication(sys.argv)
    
    # Get the directory of the executable or script
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        app_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for splash image
    splash_image = os.path.join(app_dir, 'resources', 'splash.png')
    if not os.path.exists(splash_image):
        splash_image = os.path.join(app_dir, 'splash.png')
        if not os.path.exists(splash_image):
            return None
    
    # Create splash screen
    pixmap = QPixmap(splash_image)
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    
    return splash, app

if __name__ == "__main__":
    # This is for testing the splash screen independently
    splash, app = show_splash()
    
    # Close the splash after 3 seconds
    QTimer.singleShot(3000, splash.close)
    
    sys.exit(app.exec())
"""

    with open(splash_script, "w") as f:
        f.write(content)
    
    print(f"Created splash screen script: {splash_script}")
    return splash_script

def run_pyinstaller(spec_file):
    """Run PyInstaller with the spec file"""
    cmd = f"pyinstaller --clean {spec_file}"
    print(f"Running PyInstaller: {cmd}")
    os.system(cmd)

def main():
    # Create splash screen script
    create_splash_screen_script()
    
    # Find resource files
    data_files = find_resource_files()
    
    # Create spec file
    spec_file = create_spec_file(data_files)
    
    # Install PyInstaller if not already installed
    os.system("pip install pyinstaller")
    
    # Run PyInstaller
    run_pyinstaller(spec_file)
    
    # Output success message
    print("\n" + "="*50)
    print("Build complete! Your executable is in the dist/ folder.")
    print("="*50)

if __name__ == "__main__":
    main()
