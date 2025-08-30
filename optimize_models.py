#!/usr/bin/env python3

"""
Check and optimize OpenVINO models to FP16 precision.
This script checks if the models are using FP16 precision and converts them if needed.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def check_model_precision(model_path):
    """
    Check if the model is using FP16 precision.
    
    Args:
        model_path: Path to the model XML file
        
    Returns:
        Tuple of (is_fp16, num_fp32_layers, num_total_layers)
    """
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found!")
        return False, 0, 0
        
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    fp32_layers = 0
    total_layers = 0
    
    # Check layers precision
    for layer in root.findall(".//layer"):
        total_layers += 1
        precision = layer.get("precision")
        if precision == "FP32":
            fp32_layers += 1
    
    is_fp16 = fp32_layers == 0
    
    return is_fp16, fp32_layers, total_layers

def convert_to_fp16(model_path):
    """
    Convert OpenVINO model to FP16 precision.
    
    Args:
        model_path: Path to the model XML file
        
    Returns:
        Path to the converted model
    """
    try:
        from openvino.tools import mo
        
        print(f"🔄 Converting model to FP16: {model_path}")
        
        # Get paths
        xml_path = Path(model_path)
        bin_path = xml_path.with_suffix('.bin')
        output_dir = xml_path.parent
        
        if not xml_path.exists() or not bin_path.exists():
            print(f"❌ Model files not found: {xml_path} or {bin_path}")
            return None
            
        # Run model optimizer to convert to FP16
        args = [
            "--input_model", str(xml_path),
            "--output_dir", str(output_dir),
            "--data_type", "FP16"
        ]
        
        print(f"⚙️ Running Model Optimizer with args: {args}")
        start_time = time.time()
        mo.main(args)
        conversion_time = time.time() - start_time
        
        print(f"✅ Model converted to FP16 in {conversion_time:.2f} seconds")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_model(model_path):
    """
    Check and optimize model to FP16 precision if needed.
    
    Args:
        model_path: Path to the model XML file
        
    Returns:
        Path to the optimized model
    """
    if not Path(model_path).exists():
        print(f"❌ Model file {model_path} not found!")
        return None
        
    print(f"🔍 Checking model precision: {model_path}")
    is_fp16, fp32_layers, total_layers = check_model_precision(model_path)
    
    if is_fp16:
        print(f"✅ Model is already using FP16 precision: {model_path}")
        return model_path
    else:
        print(f"⚠️ Model using FP32 precision ({fp32_layers}/{total_layers} layers). Converting to FP16...")
        return convert_to_fp16(model_path)

def main():
    """
    Check and optimize all OpenVINO models in the workspace.
    """
    print("\n" + "="*80)
    print("OpenVINO Model Optimizer - FP32 to FP16 Converter")
    print("="*80)
    
    # Check for OpenVINO
    try:
        import openvino as ov
        print(f"✅ OpenVINO version: {ov.__version__}")
    except ImportError:
        print("⚠️ OpenVINO not installed. Installing now...")
        os.system('pip install --quiet "openvino>=2024.0.0"')
        import openvino as ov
        print(f"✅ OpenVINO installed: {ov.__version__}")
    
    # Find OpenVINO models
    search_dirs = [
        ".",
        "openvino_models",
        "models",
        "../openvino_models"
    ]
    
    print("🔍 Searching for OpenVINO models...")
    
    models_found = []
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
            
        # Find XML files
        for xml_file in search_path.glob("**/*.xml"):
            if "openvino" in str(xml_file).lower() or "yolo" in str(xml_file).lower():
                models_found.append(xml_file)
    
    if not models_found:
        print("❌ No OpenVINO models found!")
        return
    
    print(f"✅ Found {len(models_found)} OpenVINO models:")
    for i, model_path in enumerate(models_found):
        print(f"   {i+1}. {model_path}")
    
    # Process each model
    optimized_models = []
    for model_path in models_found:
        optimized_path = optimize_model(model_path)
        if optimized_path:
            optimized_models.append(optimized_path)
    
    print(f"\n✅ Optimized {len(optimized_models)} models")

if __name__ == "__main__":
    main()
