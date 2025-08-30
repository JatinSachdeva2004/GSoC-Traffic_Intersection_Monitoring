"""
Prepare environment for enhanced controller.
This script checks and fixes import paths before running the main application.
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_and_fix_paths():
    """Check and fix import paths for enhanced controller."""
    print("\n" + "="*80)
    print("🔧 Checking and fixing import paths for enhanced controller")
    print("="*80)
    
    # Get the current working directory
    current_dir = Path.cwd()
    project_root = Path(__file__).parent.parent
    
    # Add necessary paths to sys.path
    paths_to_add = [
        str(project_root),
        str(project_root / "qt_app_pyside"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            print(f"Adding path to sys.path: {path}")
            sys.path.append(path)
      # Check for enhanced_annotation_utils
    try:
        print("Checking for enhanced_annotation_utils...")
        from qt_app_pyside.utils.enhanced_annotation_utils import enhanced_cv_to_qimage
        print("✅ Enhanced annotation utils found")
    except ImportError:
        print("⚠️ Could not import enhanced_annotation_utils")
        
        # Create __init__.py if missing
        init_path = project_root / "qt_app_pyside" / "utils" / "__init__.py"
        if not init_path.exists():
            print(f"Creating missing __init__.py at {init_path}")
            with open(init_path, 'w') as f:
                f.write('"""Utils package initialization"""\n')
        
        # Check for fallback annotation utils
        fallback_path = project_root / "fallback_annotation_utils.py"
        if fallback_path.exists():
            print(f"✅ Fallback annotation utils found at {fallback_path}")
        else:
            # Create a minimal fallback_annotation_utils.py
            print(f"❌ Fallback annotation utils not found, creating minimal version")
            fallback_content = '''"""Minimal fallback annotation utilities"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

def enhanced_draw_detections(frame, detections, **kwargs):
    """Minimal implementation that just returns the frame"""
    return frame

def draw_performance_overlay(frame, metrics):
    """Minimal implementation that just returns the frame"""
    return frame
    
def enhanced_cv_to_qimage(frame):
    """Minimal implementation that returns None"""
    return None
    
def enhanced_cv_to_pixmap(frame):
    """Minimal implementation that returns None"""
    return None
'''
            with open(fallback_path, 'w') as f:
                f.write(fallback_content)
            print(f"✅ Created minimal fallback annotation utils at {fallback_path}")
    
    # Check for detection_openvino_async
    try:
        print("Checking for detection_openvino_async...")
        module_path = project_root / "detection_openvino_async.py"
        if module_path.exists():
            print(f"✅ detection_openvino_async.py found at {module_path}")
        else:
            print(f"❌ detection_openvino_async.py not found at {module_path}")
    except Exception as e:
        print(f"❌ Error checking for detection_openvino_async: {e}")
    
    print("\n✅ Path checking complete")
    print("="*80)

if __name__ == "__main__":
    check_and_fix_paths()
    
    # Run the main application as a subprocess (most reliable method)
    print("Starting main application...")
    app_main_path = Path(__file__).parent / "main.py"
    
    if app_main_path.exists():
        print(f"Running {app_main_path}")
        try:
            # Always use subprocess to avoid encoding issues
            import subprocess
            result = subprocess.run([sys.executable, str(app_main_path)], check=True)
            sys.exit(result.returncode)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running main.py: {e}")
            sys.exit(e.returncode)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)
    else:
        print(f"❌ Main application not found at {app_main_path}")
        sys.exit(1)
