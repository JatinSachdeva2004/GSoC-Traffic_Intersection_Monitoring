
# Smart Intersection Monitoring 1.1.0 Installation Guide

## Overview
Smart Intersection Monitoring 1.1.0 is a cross-platform, AI-powered traffic monitoring and analytics dashboard. It features real-time video detection, multi-camera support, advanced analytics, and violation detection using OpenVINO, YOLO, and modern tracking algorithms. This guide provides step-by-step instructions for installing and running Smart Intersection Monitoring on Windows and macOS.

---

## Prerequisites

### All Platforms
- **Python**: Version 3.8 or higher
- **pip**: Python package manager
- **Hardware**: CPU or GPU (Intel/AMD/NVIDIA)
- **Disk Space**: 2GB+ recommended (models and dependencies)

### Windows
- Windows 10 or higher
- Visual Studio Build Tools (for some dependencies)
- Administrator privileges (for system-wide installation)

### macOS
- macOS 10.14 or higher
- Xcode Command Line Tools (`xcode-select --install`)
- Administrator privileges (for system-wide installation)

---

## Quick Start

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd qt_app_pyside1
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   - **Recommended**: Enhanced controller
     ```bash
     python run_app.py
     ```
   - **Standard**:
     ```bash
     python main.py
     ```

---

## Building Standalone Executables

Smart Intersection Monitoring can be built into standalone executables for easy distribution.

### 1. Run the Build Script
   ```bash
   python build_crossplatform.py
   ```
   - Detects your platform automatically
   - Builds debug and release versions
   - Creates installer scripts in the `dist/` directory

### 2. Build Outputs
- **Windows**:
  - `dist/TrafficMonitor.exe` (Release)
  - `dist/TrafficMonitorDebug.exe` (Debug)
  - `install_trafficmonitor_windows.bat` (Installer)
- **macOS**:
  - `dist/TrafficMonitor.app` (Release)
  - `dist/TrafficMonitorDebug.app` (Debug)
  - `install_trafficmonitor_macos.sh` (Installer)

---

## Testing Your Build

### Windows
1. Test debug version: `dist\TrafficMonitorDebug.exe`
2. Test release version: `dist\TrafficMonitor.exe`
3. For system install: `install_trafficmonitor_windows.bat` (run as administrator)

### macOS
1. Test debug version: `open dist/TrafficMonitorDebug.app`
2. Test release version: `open dist/TrafficMonitor.app`
3. For system install: `./install_trafficmonitor_macos.sh`

---

## Configuration
- All parameters are in `config.json` (detection, violations, display, performance)
- Edit before building for environment-specific settings

---

## Features
- Real-time video detection (OpenVINO, YOLO)
- Multi-camera, webcam, RTSP, video/image support
- Live overlays: bounding boxes, labels, violations
- Analytics: trends, histograms, summary cards
- Export: CSV/JSON, config editor
- Modern UI: QSS, icons, dark/light mode
- Performance overlay: CPU, RAM, FPS
- Violation detection: red light, speed, stop sign, lane
- VLM backend: Vision-Language Model support

---

## Troubleshooting
- **Module not found**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
- **Large executable**: Remove unused models, use `--exclude-module` in PyInstaller
- **Slow startup**: First launch may be slower due to model loading
- **Permission errors (macOS)**: `sudo xattr -rd com.apple.quarantine dist/TrafficMonitor.app`
- **Import errors**: Use `python run_app.py` for correct import paths
- **Missing models/resources**: Check `models/` and `resources/` directories

---

## Advanced: Manual Build (PyInstaller)

### Windows
```bash
pyinstaller --name=TrafficMonitor --windowed --onefile --icon=resources/icon.ico --add-data="ui;ui" --add-data="controllers;controllers" main.py
```

### macOS
```bash
pyinstaller --name=TrafficMonitor --windowed --onefile --icon=resources/icon.icns --add-data="ui:ui" --add-data="controllers:controllers" --target-arch=universal2 main.py
```

---

## Support
- Check console output for errors (debug version)
- Verify all dependencies and permissions
- For development, use debug builds for detailed logs
- For further help, consult the user guide or contact the development team

---

## VersionGuard 1.1.0 Release Notes
- Enhanced async inference pipeline
- FP16 precision for CPU
- Auto model selection (CPU/GPU)
- Optimized DeepSORT tracking
- Improved analytics and export
- Cross-platform installer scripts

---

*Thank you for using Smart Intersection Monitoring!*
