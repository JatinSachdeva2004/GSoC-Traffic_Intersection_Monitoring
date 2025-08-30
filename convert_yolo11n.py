#!/usr/bin/env python3

import os
import sys
import time
import shutil
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the conversion function from detection_openvino.py
from detection_openvino import convert_yolo_to_openvino

def main():
    """
    Convert yolo11n.pt model to OpenVINO IR format.
    Usage: python convert_yolo11n.py
    """
    print("\n" + "="*80)
    print("YOLO11n Model Converter - PyTorch to OpenVINO IR")
    print("="*80)
      # Check if the model exists
    model_path = Path("yolo11n.pt")
    if not model_path.exists():
        print(f"❌ Error: Model file {model_path} not found!")
        print(f"   Please ensure '{model_path}' is in the current directory.")
        return
    
    print(f"✅ Found model: {model_path}")
    
    # Check for OpenVINO and other dependencies
    try:
        import openvino as ov
        print(f"✅ OpenVINO version: {ov.__version__}")
    except ImportError:
        print("⚠️ OpenVINO not installed. Installing now...")
        os.system('pip install --quiet "openvino>=2024.0.0"')
        import openvino as ov
        print(f"✅ OpenVINO installed: {ov.__version__}")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("⚠️ Ultralytics not installed. Installing now...")
        os.system('pip install --quiet "ultralytics>=8.0.0"')
        from ultralytics import YOLO
        print("✅ Ultralytics installed")
    
    # Create destination directory for the models
    openvino_dir = Path("openvino_models")
    if not openvino_dir.exists():
        openvino_dir.mkdir(exist_ok=True)
        print(f"✅ Created directory: {openvino_dir}")
    
    try:
        # Convert model to OpenVINO IR format
        print("\n📦 Converting model to OpenVINO IR format...")
        start_time = time.time()
        output_path = convert_yolo_to_openvino("yolo11n", half=True)
        conversion_time = time.time() - start_time
        
        print(f"✅ Conversion completed in {conversion_time:.2f} seconds!")
        print(f"✅ Output model: {output_path}")
        
        # Verify output files
        if output_path and Path(output_path).exists():
            xml_path = Path(output_path)
            bin_path = xml_path.with_suffix('.bin')
            xml_size = xml_path.stat().st_size / (1024 * 1024)  # in MB
            bin_size = bin_path.stat().st_size / (1024 * 1024)  # in MB
            
            print(f"✅ XML file: {xml_path} ({xml_size:.2f} MB)")
            print(f"✅ BIN file: {bin_path} ({bin_size:.2f} MB)")
            
            # Copy to openvino_models directory for easier access by the Qt app
            dst_xml = openvino_dir / xml_path.name
            dst_bin = openvino_dir / bin_path.name
            
            shutil.copy2(xml_path, dst_xml)
            shutil.copy2(bin_path, dst_bin)
            
            print(f"✅ Copied models to: {openvino_dir}")
            print("\n🚀 Model conversion and setup complete!")
            print("\n📋 Instructions:")
            print(f"   1. The model files are available at: {openvino_dir}")
            print("   2. In the Qt app, you can now select this model from the dropdown")
            print("   3. Use the device selection dropdown to choose between CPU and GPU")
        else:
            print("❌ Failed to verify output files.")
    
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
