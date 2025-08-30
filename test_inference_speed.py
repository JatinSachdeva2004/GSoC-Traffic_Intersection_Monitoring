#!/usr/bin/env python3

"""
Test OpenVINO inference speed with different models and devices.
This script helps you benchmark the performance of YOLO models on different devices.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the needed modules
try:
    import openvino as ov
except ImportError:
    print("Installing openvino...")
    os.system('pip install --quiet "openvino>=2024.0.0"')
    import openvino as ov

def test_model_inference(model_path, device="AUTO", num_iterations=100):
    """
    Test model inference speed.
    
    Args:
        model_path: Path to the model XML file
        device: Device to run inference on (CPU, GPU, AUTO)
        num_iterations: Number of iterations for the test
        
    Returns:
        Dict with performance metrics
    """
    print(f"\n🔍 Testing model: {model_path} on device: {device}")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return None
        
    # Load model
    try:
        core = ov.Core()
        model = core.read_model(model_path)
        
        # Configure model
        ov_config = {}
        if device != "CPU":
            model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
            
        # Compile model
        print(f"⚙️ Compiling model for {device}...")
        compiled_model = core.compile_model(model=model, device_name=device, config=ov_config)
          # Create dummy input - handle dynamic shapes properly
        try:
            # For dynamic models, we need to use explicit shape
            dummy_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
            print(f"Using explicit input shape: (1, 3, 640, 640)")
        except Exception as e:
            print(f"Error creating dummy input: {e}")
            return None
        
        # Warm up
        print("🔥 Warming up...")
        for _ in range(10):
            compiled_model(dummy_input)
            
        # Run inference
        print(f"⏱️ Running {num_iterations} iterations...")
        inference_times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            output = compiled_model(dummy_input)[0]
            inference_time = time.time() - start_time
            inference_times.append(inference_time * 1000)  # Convert to ms
            
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}/{num_iterations}, time: {inference_time * 1000:.2f} ms")
                
        # Calculate statistics
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_dev = np.std(inference_times)
        fps = 1000 / avg_time
        
        print("\n📊 Results:")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Min inference time: {min_time:.2f} ms")
        print(f"  Max inference time: {max_time:.2f} ms")
        print(f"  Standard deviation: {std_dev:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        return {
            "model": model_path,
            "device": device,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_dev_ms": std_dev,
            "fps": fps
        }
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def find_models():
    """
    Find all OpenVINO models in the workspace.
    
    Returns:
        List of model paths
    """
    search_dirs = [
        ".",
        "openvino_models",
        "models",
        "../openvino_models"
    ]
    
    models_found = []
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        if not search_path.exists():
            continue
            
        # Find XML files
        for xml_file in search_path.glob("**/*.xml"):
            if "openvino" in str(xml_file).lower() or "yolo" in str(xml_file).lower():
                models_found.append(xml_file)
    
    return models_found

def validate_device_safely(core, device, model_path):
    """
    Safely validate if a device can actually run inference.
    
    Args:
        core: OpenVINO core object
        device: Device name to test  
        model_path: Path to model for testing
        
    Returns:
        bool: True if device works, False otherwise
    """
    try:
        print(f"🔍 Testing device {device}...")
        
        # Try to read and compile model
        model = core.read_model(model_path)
        compiled_model = core.compile_model(model, device)
        
        # Try a simple inference
        dummy_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
        result = compiled_model(dummy_input)
        
        print(f"✅ Device {device} works!")
        return True
        
    except Exception as e:
        print(f"❌ Device {device} failed: {str(e)[:100]}...")
        return False

def main():
    """
    Main entry point.
    """
    print("\n" + "="*80)
    print("OpenVINO Model Inference Speed Test")
    print("="*80)
    
    # Check available devices with proper validation
    core = ov.Core()
    raw_devices = core.available_devices
    print(f"🔍 Raw available devices: {raw_devices}")
    
    # Validate which devices actually work
    available_devices = ["CPU"]  # CPU always works
    
    # Test GPU availability
    if "GPU" in raw_devices:
        try:
            # Try to create a simple model on GPU
            test_model = core.read_model("openvino_models/yolo11n.xml") if Path("openvino_models/yolo11n.xml").exists() else None
            if test_model:
                gpu_compiled = core.compile_model(test_model, "GPU")
                test_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
                gpu_compiled(test_input)  # Try one inference
                available_devices.append("GPU")
                print("✅ GPU validation successful")
            else:
                print("⚠️ No model found for GPU validation")
        except Exception as e:
            print(f"❌ GPU validation failed: {e}")
    
    # Test NPU availability 
    if "NPU" in raw_devices:
        try:
            test_model = core.read_model("openvino_models/yolo11n.xml") if Path("openvino_models/yolo11n.xml").exists() else None
            if test_model:
                npu_compiled = core.compile_model(test_model, "NPU")
                test_input = np.random.random((1, 3, 640, 640)).astype(np.float32)
                npu_compiled(test_input)  # Try one inference
                available_devices.append("NPU")
                print("✅ NPU validation successful")
        except Exception as e:
            print(f"❌ NPU validation failed: {e}")
    
    print(f"✅ Validated working devices: {available_devices}")
    
    # Find models
    models = find_models()
    if not models:
        print("❌ No models found!")
        return
        
    print(f"✅ Found {len(models)} models:")
    for i, model_path in enumerate(models):
        print(f"   {i+1}. {model_path}")
          # Find the best model for CPU testing (prefer yolo11n)
    yolo11n_idx = -1
    for idx, model_path in enumerate(models):
        if "yolo11n" in str(model_path).lower() and "openvino_models" in str(model_path).lower():
            yolo11n_idx = idx
            break
    
    if yolo11n_idx == -1:
        for idx, model_path in enumerate(models):
            if "yolo11n" in str(model_path).lower():
                yolo11n_idx = idx
                break
    
    # Set default model to yolo11n if found, otherwise use first model
    model_idx = yolo11n_idx if yolo11n_idx != -1 else 0
    
    # Allow user to override if desired
    print("\nRecommended model for CPU: " + str(models[model_idx]))
    try:
        user_input = input("Press Enter to use recommended model or enter a number to choose different model: ")
        if user_input.strip():
            user_idx = int(user_input) - 1
            if 0 <= user_idx < len(models):
                model_idx = user_idx
    except (ValueError, IndexError):
        pass  # Keep the default/recommended model
        
    selected_model = models[model_idx]
    print(f"✅ Selected model: {selected_model}")
    
    # Test on all available devices
    results = []
    for device in available_devices:
        result = test_model_inference(selected_model, device)
        if result:
            results.append(result)
            
    # Print comparison
    if len(results) > 1:
        print("\n📊 Device Comparison:")
        print("-" * 80)
        print(f"{'Device':<10} {'Avg Time (ms)':<15} {'Min Time (ms)':<15} {'Max Time (ms)':<15} {'FPS':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['device']:<10} {result['avg_time_ms']:<15.2f} {result['min_time_ms']:<15.2f} {result['max_time_ms']:<15.2f} {result['fps']:<10.2f}")
            
        print("\n🏆 Fastest device: " + max(results, key=lambda x: x['fps'])['device'])
    
if __name__ == "__main__":
    main()
