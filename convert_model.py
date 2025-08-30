#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install --quiet "ultralytics>=8.0.0"')
    from ultralytics import YOLO

def convert_pt_to_openvino(model_path: str, output_dir: str = None, half: bool = False):
    """
    Convert PyTorch model to OpenVINO IR format.
    
    Args:
        model_path: Path to PyTorch .pt model file
        output_dir: Directory to save converted model (default is same as model with _openvino_model suffix)
        half: Whether to use half precision (FP16)
    
    Returns:
        Path to the converted XML file
    """
    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Get model name without extension for output directory
    model_name = model_path.stem
    
    # Set output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        # We'll still use model_name for the file names
    else:
        output_dir = model_path.parent / f"{model_name}_openvino_model"
    
    ov_xml = output_dir / f"{model_name}.xml"
    
    # Check if model already exists
    if ov_xml.exists():
        print(f"OpenVINO model already exists: {ov_xml}")
        print(f"To reconvert, delete or rename the existing files.")
        return str(ov_xml)
    
    # Load model and export
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"Exporting to OpenVINO IR format...")
    print(f"Output directory: {output_dir}")
    print(f"Using half precision: {half}")
    
    # Export the model (will create both .xml and .bin files)
    model.export(format="openvino", dynamic=True, half=half, imgsz=640)
    
    # Verify files were created
    if ov_xml.exists():
        print(f"✅ Conversion successful!")
        print(f"XML file: {ov_xml}")
        print(f"BIN file: {ov_xml.with_suffix('.bin')}")
        return str(ov_xml)
    else:
        print(f"❌ Conversion failed - output files not found")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO PyTorch models to OpenVINO IR format")
    parser.add_argument("model_path", type=str, help="Path to PyTorch .pt model file")
    parser.add_argument("--output", type=str, default=None, help="Directory to save converted model")
    parser.add_argument("--half", action="store_true", help="Use half precision (FP16)")
    
    args = parser.parse_args()
    
    convert_pt_to_openvino(args.model_path, args.output, args.half)
