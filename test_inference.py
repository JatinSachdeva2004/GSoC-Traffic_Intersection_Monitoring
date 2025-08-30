"""
YOLOv11 OpenVINO Benchmark Suite
Covers:
1. YOLOv11n vs YOLOv11x on CPU/GPU: Latency, Jitter, Spikes
2. INT8 Quantized YOLOv11: FP32 vs INT8 (Speed, Accuracy, Size)
3. Parallel Inference: Multiple Models on Shared CPU/GPU
4. Power Efficiency: FPS/Watt for YOLOv11 Variants
5. Graph Optimization Logs for YOLOv11x
"""
import os
import time
import numpy as np
from openvino.runtime import Core
import threading
import psutil

# --------- CONFIG ---------
MODEL_PATHS = {
    # YOLOv11n models
    'yolov11n': 'openvino_models/yolo11n.xml',
    'yolov11n_root': 'yolo11n_openvino_model/yolo11n.xml',
    
    # YOLOv11x models
    'yolov11x_root': 'yolo11x.xml',  # Root directory
    'yolov11x_folder': 'yolo11x_openvino_model/yolo11x.xml',
    'yolov11x_models': 'models/yolo11x_openvino_model/yolo11x.xml',
    
    # Placeholders for INT8 models (if they exist)
    'yolov11n_int8': 'openvino_models/yolo11n_int8.xml',
    'yolov11x_int8': 'openvino_models/yolo11x_int8.xml',
}

# Verify which models actually exist and create filtered dictionaries
def get_available_models(model_dict):
    """Returns a dictionary of only the models that actually exist in the filesystem"""
    available_models = {}
    for key, path in model_dict.items():
        if os.path.exists(path):
            available_models[key] = path
    return available_models

def get_models_by_type(model_dict, model_type):
    """Returns a dictionary of models that match a specific type (e.g., 'yolo11n' or 'yolo11x')"""
    return {k: v for k, v in model_dict.items() if model_type in k}
SAMPLE_IMAGE = 'sample.jpg'  # Place a sample image in root or update path
DEVICE_LIST = ['CPU', 'GPU']
N_PARALLEL = 2  # For parallel inference

# --------- UTILS ---------
def load_model(core, model_path, device):
    compiled_model = core.compile_model(model_path, device)
    return compiled_model

def run_inference(compiled_model, input_image, n_iter=50):
    infer_times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        _ = compiled_model([input_image])
        infer_times.append((time.perf_counter() - start) * 1000)
    return np.array(infer_times)

def print_latency_stats(times, label):
    print(f"\n{label}")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std (Jitter): {np.std(times):.2f} ms")
    print(f"  Max (Spike): {np.max(times):.2f} ms")
    print(f"  Min: {np.min(times):.2f} ms")

# --------- 1. Latency & Stability ---------
def test_latency_stability():
    print("\n=== 1. YOLOv11n vs YOLOv11x Latency & Stability ===")
    core = Core()
    img = np.ones((1, 3, 640, 640), dtype=np.float32)  # Dummy input
    
    # Get available models
    available_models = get_available_models(MODEL_PATHS)
    if not available_models:
        print("No models found for testing. Please check MODEL_PATHS.")
        return
        
    # Get YOLOv11n and YOLOv11x models separately for comparison
    yolo11n_models = get_models_by_type(available_models, 'yolo11n')
    yolo11x_models = get_models_by_type(available_models, 'yolo11x')
    
    print(f"Found {len(yolo11n_models)} YOLOv11n models and {len(yolo11x_models)} YOLOv11x models.")
    
    # Test all available models on all devices
    for device in DEVICE_LIST:
        # First test nano models
        for model_key, model_path in yolo11n_models.items():
            print(f"\nTesting {model_key} ({model_path}) on {device}")
            try:
                model = load_model(core, model_path, device)
                times = run_inference(model, img)
                print_latency_stats(times, f"{model_key} on {device}")
            except Exception as e:
                print(f"Error loading or running {model_key} on {device}: {e}")
        
        # Then test x models
        for model_key, model_path in yolo11x_models.items():
            print(f"\nTesting {model_key} ({model_path}) on {device}")
            try:
                model = load_model(core, model_path, device)
                times = run_inference(model, img)
                print_latency_stats(times, f"{model_key} on {device}")
            except Exception as e:
                print(f"Error loading or running {model_key} on {device}: {e}")

# --------- 2. INT8 Quantization ---------
def test_int8_quantization():
    print("\n=== 2. INT8 Quantization: FP32 vs INT8 ===")
    core = Core()
    img = np.ones((1, 3, 640, 640), dtype=np.float32)
    
    # Get available models
    available_models = get_available_models(MODEL_PATHS)
    
    # Check if we have INT8 models
    int8_models = {k: v for k, v in available_models.items() if 'int8' in k.lower()}
    fp32_models = {k: v for k, v in available_models.items() if 'int8' not in k.lower()}
    
    if not int8_models:
        print("No INT8 models found. Comparing only FP32 models.")
        
    # Group models by type for comparison
    model_groups = {}
    for model_key, model_path in available_models.items():
        base_type = 'yolo11n' if 'yolo11n' in model_key else 'yolo11x'
        if base_type not in model_groups:
            model_groups[base_type] = []
        model_groups[base_type].append((model_key, model_path))
    
    # Process each model group
    for base_type, models in model_groups.items():
        print(f"\n--- {base_type.upper()} Models ---")
        for model_key, model_path in models:
            print(f"\nTesting {model_key} ({model_path}) on CPU")
            try:
                model = load_model(core, model_path, 'CPU')
                times = run_inference(model, img)
                print_latency_stats(times, f"{model_key} on CPU")
                
                # Model size (XML and BIN)
                xml_size = os.path.getsize(model_path) / (1024*1024)
                bin_path = model_path.replace('.xml', '.bin')
                bin_size = os.path.getsize(bin_path) / (1024*1024)
                total_size = xml_size + bin_size
                print(f"  Model size (XML): {xml_size:.2f} MB")
                print(f"  Model size (BIN): {bin_size:.2f} MB")
                print(f"  Total model size: {total_size:.2f} MB")
                
                # Precision info
                print(f"  Precision: {'INT8' if 'int8' in model_key.lower() else 'FP32'}")
                
                # TODO: Add accuracy eval if ground truth available
                # print("  Accuracy: <requires ground truth data>")
            except Exception as e:
                print(f"Error testing {model_key}: {e}")
                
    # Summary of INT8 vs FP32 comparison
    if int8_models and fp32_models:
        print("\n--- INT8 vs FP32 Summary ---")
        print("Model type | Precision | Avg Latency | Size | Recommended for")
        print("-----------------------------------------------------------")
        # This would be populated with actual data from tests
        print("This comparison requires running the above tests and collecting results.")
        print("INT8 models typically offer 2-4x speedup with 5-10% accuracy loss and 75% size reduction.")

# --------- 3. Parallel Inference ---------
def parallel_worker(model_path, device, img, results, idx):
    try:
        core = Core()
        model = load_model(core, model_path, device)
        times = run_inference(model, img, n_iter=20)  # Reduce iterations for parallel test
        results[idx] = times
    except Exception as e:
        print(f"Error in worker thread {idx} with {model_path} on {device}: {e}")
        results[idx] = None

def test_parallel_inference():
    print("\n=== 3. Parallel Inference: Multiple Models on Shared Device ===")
    img = np.ones((1, 3, 640, 640), dtype=np.float32)
    
    # Get available models
    available_models = get_available_models(MODEL_PATHS)
    if not available_models:
        print("No models found for parallel testing")
        return
    
    # Test different scenarios:
    # 1. Multiple instances of same model
    # 2. Different models in parallel (if we have both nano and x)
    
    # Get one YOLOv11n and one YOLOv11x model if available
    yolo11n_models = get_models_by_type(available_models, 'yolo11n')
    yolo11x_models = get_models_by_type(available_models, 'yolo11x')
    
    # Single model parallel test
    for device in DEVICE_LIST:
        print(f"\n--- Testing parallel instances on {device} ---")
        
        # Test each model type
        for model_dict in [yolo11n_models, yolo11x_models]:
            if not model_dict:
                continue
                
            # Take the first model from each type
            model_key = list(model_dict.keys())[0]
            model_path = model_dict[model_key]
            
            print(f"\nRunning {N_PARALLEL} parallel instances of {model_key} ({model_path}) on {device}")
            threads = []
            results = [None] * N_PARALLEL
            
            for i in range(N_PARALLEL):
                t = threading.Thread(target=parallel_worker, args=(model_path, device, img, results, i))
                threads.append(t)
                t.start()
                
            for t in threads:
                t.join()
                
            # Calculate combined stats
            all_times = []
            for i, times in enumerate(results):
                if times is not None:
                    print_latency_stats(times, f"Thread {i+1} {model_key} on {device}")
                    all_times.extend(times)
                else:
                    print(f"Thread {i+1} failed for {model_key} on {device}")
            
            if all_times:
                print(f"\nCombined statistics for parallel {model_key} instances:")
                print(f"  Total inferences: {len(all_times)}")
                print(f"  Aggregate FPS: {len(all_times)/sum(all_times)*1000:.2f}")
    
    # Mixed model parallel test (if we have both nano and x models)
    if yolo11n_models and yolo11x_models:
        print("\n--- Testing different models in parallel ---")
        for device in DEVICE_LIST:
            print(f"\nMixing YOLOv11n and YOLOv11x on {device}")
            
            nano_key = list(yolo11n_models.keys())[0]
            x_key = list(yolo11x_models.keys())[0]
            
            threads = []
            results = [None] * 2
            model_keys = [nano_key, x_key]
            model_paths = [yolo11n_models[nano_key], yolo11x_models[x_key]]
            
            for i in range(2):
                t = threading.Thread(target=parallel_worker, args=(model_paths[i], device, img, results, i))
                threads.append(t)
                t.start()
                
            for t in threads:
                t.join()
                
            for i, times in enumerate(results):
                if times is not None:
                    print_latency_stats(times, f"{model_keys[i]} on {device} (mixed mode)")
                else:
                    print(f"{model_keys[i]} failed on {device} (mixed mode)")

# --------- 4. Power Efficiency ---------
def test_power_efficiency():
    print("\n=== 4. Power Efficiency: FPS/Watt ===")
    # NOTE: This requires external power measurement (e.g., RAPL, nvidia-smi, or a power meter)
    # Here, we just print FPS and leave a TODO for power measurement
    core = Core()
    img = np.ones((1, 3, 640, 640), dtype=np.float32)
    
    # Use the models we know exist
    models_to_test = []
    for model_key in MODEL_PATHS:
        if os.path.exists(MODEL_PATHS[model_key]):
            models_to_test.append(model_key)
    
    if not models_to_test:
        print("No models found for power efficiency testing")
        return
        
    print("\nModels to test:", models_to_test)
    
    for model_key in models_to_test:
        try:
            print(f"\nTesting {model_key} ({MODEL_PATHS[model_key]}) on CPU")
            model = load_model(core, MODEL_PATHS[model_key], 'CPU')
            start = time.perf_counter()
            n_iter = 100
            for _ in range(n_iter):
                _ = model([img])
            elapsed = time.perf_counter() - start
            fps = n_iter / elapsed
            
            # Try to estimate power using psutil (very rough estimate)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            print(f"{model_key} on CPU: {fps:.2f} FPS (CPU load: {cpu_percent}%)")
        except Exception as e:
            print(f"Error testing power efficiency for {model_key}: {e}")
            
    print("\nFor accurate power measurements:")
    print("- On Linux: Use RAPL via 'intel-power-gadget' or '/sys/class/powercap/intel-rapl'")
    print("- On Windows: Use Intel Power Gadget, HWiNFO, or an external power meter")
    print("- For NVIDIA GPUs: Use 'nvidia-smi' to monitor power consumption")

# --------- 5. Graph Optimization Logs ---------
def test_graph_optimization_logs():
    print("\n=== 5. OpenVINO Graph Optimization Logs for YOLOv11x ===")
    
    # Try each available YOLOv11x model
    yolo_models = [key for key in MODEL_PATHS.keys() if 'yolo11x' in key and os.path.exists(MODEL_PATHS[key])]
    
    if not yolo_models:
        print("No YOLOv11x models found for graph optimization analysis")
        return
        
    # Use the first available YOLOv11x model
    model_key = yolo_models[0]
    model_path = MODEL_PATHS[model_key]
    
    print(f"Using {model_key} ({model_path}) for graph analysis")
    
    try:
        core = Core()
        # Enable OpenVINO debug logs
        os.environ['OV_DEBUG_LOG_LEVEL'] = 'DEBUG'
        print("Compiling model with debug logs...")
        model = load_model(core, model_path, 'CPU')
        
        # Print model ops
        print("\nModel operations:")
        ops = list(model.model.get_ops())
        print(f"Total operations: {len(ops)}")
        
        # Group operations by type
        op_types = {}
        for op in ops:
            op_type = op.get_type_name()
            if op_type not in op_types:
                op_types[op_type] = 0
            op_types[op_type] += 1
        
        # Print operation types summary
        print("\nOperation types summary:")
        for op_type, count in sorted(op_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {op_type}: {count} ops")
        
        # Print first 10 operations in detail
        print("\nSample operations (first 10):")
        for i, op in enumerate(ops[:10]):
            print(f"  {i+1}. {op.get_friendly_name()} ({op.get_type_name()})")
            
        print("\nCheck OpenVINO logs for detailed optimization info.")
    except Exception as e:
        print(f"Error analyzing model graph: {e}")

# --------- MAIN ---------
if __name__ == "__main__":
    test_latency_stability()
    test_int8_quantization()
    test_parallel_inference()
    test_power_efficiency()
    test_graph_optimization_logs()
