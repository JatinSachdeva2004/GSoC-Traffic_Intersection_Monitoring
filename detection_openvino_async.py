"""
Enhanced OpenVINO vehicle detector with async inference support
"""

# Import original detector to extend it
from detection_openvino import OpenVINOVehicleDetector as BaseDetector
import numpy as np
import time
from typing import List, Dict, Optional

class OpenVINOVehicleDetector(BaseDetector):
    """
    Enhanced OpenVINO vehicle detector with async inference support
    """
    def __init__(self, model_path: str = None, device: str = "AUTO", 
                 use_quantized: bool = False, enable_ocr: bool = False, 
                 confidence_threshold: float = 0.4, num_requests: int = 4):
        """
        Initialize the detector with async inference support.
        
        Args:
            model_path: Path to the model XML file
            device: Inference device (CPU, GPU, AUTO)
            use_quantized: Whether to use INT8 quantized model
            enable_ocr: Whether to enable OCR
            confidence_threshold: Detection confidence threshold
            num_requests: Number of async inference requests to create
        """
        # Initialize the base detector
        super().__init__(model_path, device, use_quantized, enable_ocr, confidence_threshold)
        
        # Create multiple inference requests for pipelining
        self.num_requests = num_requests
        self.infer_requests = [self.compiled_model.create_infer_request() for _ in range(num_requests)]
        self.current_request_idx = 0
        
        # Keep track of requests in flight
        self.active_requests = {}  # frame_id -> (request, frame_shape, start_time)
        self.next_frame_id = 0
        
        print(f"✅ Created {num_requests} async inference requests for {device} device")
        
    def detect_async_start(self, frame: np.ndarray) -> int:
        """
        Start asynchronous detection on a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            frame_id: ID to use when retrieving results
        """
        # Get next available request
        request = self.infer_requests[self.current_request_idx]
        self.current_request_idx = (self.current_request_idx + 1) % len(self.infer_requests)
        
        # Preprocess frame
        preprocessed_frame = self._preprocess(frame)
        
        # Get frame ID and add to active requests
        frame_id = self.next_frame_id
        self.next_frame_id += 1
        
        # Record the start time for performance tracking
        start_time = time.time()
        
        # Start async inference
        request.start_async({0: preprocessed_frame})
        
        # Store request info
        self.active_requests[frame_id] = (request, frame.shape[:2], start_time)
        
        return frame_id
        
    def detect_async_get_result(self, frame_id: int, wait: bool = True,
                               conf_threshold: Optional[float] = None) -> Optional[List[Dict]]:
        """
        Get results from an async inference request.
        
        Args:
            frame_id: Frame ID returned from detect_async_start
            wait: Whether to wait for the request to complete
            conf_threshold: Optional confidence threshold override
            
        Returns:
            Detections or None if not ready
        """
        if frame_id not in self.active_requests:
            print(f"⚠️ Frame ID {frame_id} not found in active requests")
            return None
            
        request, frame_shape, start_time = self.active_requests[frame_id]
        
        # Check if request is complete
        if wait:
            request.wait()
        elif request.wait(0) != 0:  # Not finished yet
            return None
            
        # Get output and process
        output = request.get_output_tensor().data
        
        # Use provided threshold or default
        threshold = conf_threshold if conf_threshold is not None else self.confidence_threshold
        
        # Process results
        detections = self._postprocess(output, frame_shape, threshold)
        
        # Update performance stats
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)
        if len(self._inference_times) > 30:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = np.mean(self._inference_times) * 1000
        self.performance_stats['frames_processed'] += 1
        self._frame_count += 1
        self.performance_stats['total_detections'] += len(detections)
        
        # Clean up
        del self.active_requests[frame_id]
        
        return detections
        
    def are_requests_complete(self) -> bool:
        """Check if all inference requests are complete."""
        return len(self.active_requests) == 0
        
    def wait_for_all(self) -> None:
        """Wait for all outstanding inference requests to complete."""
        for frame_id in list(self.active_requests.keys()):
            self.detect_async_get_result(frame_id, wait=True)
            
    def detect_vehicles(self, frame: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect vehicles in a frame using async API internally.
        This maintains compatibility with the existing API but uses async under the hood.
        
        Args:
            frame: Input frame
            conf_threshold: Optional confidence threshold override
            
        Returns:
            List of detections
        """
        # Start async detection
        frame_id = self.detect_async_start(frame)
        
        # Wait for and get results
        return self.detect_async_get_result(frame_id, wait=True, conf_threshold=conf_threshold)
# Detection logic using OpenVINO models (YOLO, etc.)

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from red_light_violation_pipeline import RedLightViolationPipeline

# --- Install required packages if missing ---
try:
    import openvino as ov
except ImportError:
    print("Installing openvino...")
    os.system('pip install --quiet "openvino>=2024.0.0"')
    import openvino as ov
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install --quiet "ultralytics==8.3.0"')
    from ultralytics import YOLO
try:
    import nncf
except ImportError:
    print("Installing nncf...")
    os.system('pip install --quiet "nncf>=2.9.0"')
    import nncf

# --- COCO dataset class names ---
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Traffic-related classes we're interested in (using standard COCO indices)
TRAFFIC_CLASS_NAMES = COCO_CLASSES

# --- Model Conversion and Quantization ---
def convert_yolo_to_openvino(model_name: str = "yolo11x", half: bool = True) -> Path:
    """Convert YOLOv11x PyTorch model to OpenVINO IR format."""
    pt_path = Path(f"{model_name}.pt")
    ov_dir = Path(f"{model_name}_openvino_model")
    ov_xml = ov_dir / f"{model_name}.xml"
    if not ov_xml.exists():
        print(f"Exporting {pt_path} to OpenVINO IR...")
        model = YOLO(str(pt_path))
        model.export(format="openvino", dynamic=True, half=half)
    else:
        print(f"OpenVINO IR already exists: {ov_xml}")
    return ov_xml

def quantize_openvino_model(ov_xml: Path, model_name: str = "yolo11x") -> Path:
    """Quantize OpenVINO IR model to INT8 using NNCF."""
    int8_dir = Path(f"{model_name}_openvino_int8_model")
    int8_xml = int8_dir / f"{model_name}.xml"
    if int8_xml.exists():
        print(f"INT8 model already exists: {int8_xml}")
        return int8_xml
    print("Quantization requires a calibration dataset. Skipping actual quantization in this demo.")
    return ov_xml  # Return FP32 if no quantization

# --- OpenVINO Inference Pipeline ---
class OpenVINOYOLODetector:
    def __init__(self, model_xml: Path, device: str = "AUTO"):
        self.core = ov.Core()
        self.device = device
        self.model = self.core.read_model(model_xml)
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            self.model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        input_tensor = self.preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        return self.postprocess(output, frame.shape, conf_threshold)

    def postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def run_inference_async(detector: OpenVINOVehicleDetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None, max_pipeline=4):
    """
    Run video inference using the async API of OpenVINOVehicleDetector.
    """
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Async Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    pipeline = []  # List of (frame_id, frame, t0)
    while True:
        # Fill pipeline
        while len(pipeline) < max_pipeline:
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            if video_width:
                scale = video_width / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t0 = time.time()
            frame_id = detector.detect_async_start(frame)
            pipeline.append((frame_id, frame, t0))
        if not pipeline:
            break
        # Get result for the oldest frame in pipeline
        frame_id, frame, t0 = pipeline.pop(0)
        detections = detector.detect_async_get_result(frame_id, wait=True, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - t0
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def run_inference_async(detector: OpenVINOVehicleDetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None, max_pipeline=4):
    """
    Run video inference using the async API of OpenVINOVehicleDetector.
    """
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Async Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    pipeline = []  # List of (frame_id, frame, t0)
    while True:
        # Fill pipeline
        while len(pipeline) < max_pipeline:
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            if video_width:
                scale = video_width / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t0 = time.time()
            frame_id = detector.detect_async_start(frame)
            pipeline.append((frame_id, frame, t0))
        if not pipeline:
            break
        # Get result for the oldest frame in pipeline
        frame_id, frame, t0 = pipeline.pop(0)
        detections = detector.detect_async_get_result(frame_id, wait=True, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - t0
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def run_inference_async(detector: OpenVINOVehicleDetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None, max_pipeline=4):
    """
    Run video inference using the async API of OpenVINOVehicleDetector.
    """
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Async Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    pipeline = []  # List of (frame_id, frame, t0)
    while True:
        # Fill pipeline
        while len(pipeline) < max_pipeline:
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            if video_width:
                scale = video_width / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t0 = time.time()
            frame_id = detector.detect_async_start(frame)
            pipeline.append((frame_id, frame, t0))
        if not pipeline:
            break
        # Get result for the oldest frame in pipeline
        frame_id, frame, t0 = pipeline.pop(0)
        detections = detector.detect_async_get_result(frame_id, wait=True, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - t0
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        detections = self._postprocess(output, frame.shape, conf_threshold)
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def run_inference_async(detector: OpenVINOVehicleDetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None, max_pipeline=4):
    """
    Run video inference using the async API of OpenVINOVehicleDetector.
    """
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Async Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    pipeline = []  # List of (frame_id, frame, t0)
    while True:
        # Fill pipeline
        while len(pipeline) < max_pipeline:
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)
            if video_width:
                scale = video_width / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t0 = time.time()
            frame_id = detector.detect_async_start(frame)
            pipeline.append((frame_id, frame, t0))
        if not pipeline:
            break
        # Get result for the oldest frame in pipeline
        frame_id, frame, t0 = pipeline.pop(0)
        detections = detector.detect_async_get_result(frame_id, wait=True, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - t0
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes









# Detection logic using OpenVINO models (YOLO, etc.)

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from red_light_violation_pipeline import RedLightViolationPipeline

# --- Install required packages if missing ---
try:
    import openvino as ov
except ImportError:
    print("Installing openvino...")
    os.system('pip install --quiet "openvino>=2024.0.0"')
    import openvino as ov
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system('pip install --quiet "ultralytics==8.3.0"')
    from ultralytics import YOLO
try:
    import nncf
except ImportError:
    print("Installing nncf...")
    os.system('pip install --quiet "nncf>=2.9.0"')
    import nncf

# --- COCO dataset class names ---
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Traffic-related classes we're interested in (using standard COCO indices)
TRAFFIC_CLASS_NAMES = COCO_CLASSES

# --- Model Conversion and Quantization ---
def convert_yolo_to_openvino(model_name: str = "yolo11x", half: bool = True) -> Path:
    """Convert YOLOv11x PyTorch model to OpenVINO IR format."""
    pt_path = Path(f"{model_name}.pt")
    ov_dir = Path(f"{model_name}_openvino_model")
    ov_xml = ov_dir / f"{model_name}.xml"
    if not ov_xml.exists():
        print(f"Exporting {pt_path} to OpenVINO IR...")
        model = YOLO(str(pt_path))
        model.export(format="openvino", dynamic=True, half=half)
    else:
        print(f"OpenVINO IR already exists: {ov_xml}")
    return ov_xml

def quantize_openvino_model(ov_xml: Path, model_name: str = "yolo11x") -> Path:
    """Quantize OpenVINO IR model to INT8 using NNCF."""
    int8_dir = Path(f"{model_name}_openvino_int8_model")
    int8_xml = int8_dir / f"{model_name}.xml"
    if int8_xml.exists():
        print(f"INT8 model already exists: {int8_xml}")
        return int8_xml
    print("Quantization requires a calibration dataset. Skipping actual quantization in this demo.")
    return ov_xml  # Return FP32 if no quantization

# --- OpenVINO Inference Pipeline ---
class OpenVINOYOLODetector:
    def __init__(self, model_xml: Path, device: str = "AUTO"):
        self.core = ov.Core()
        self.device = device
        self.model = self.core.read_model(model_xml)
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            self.model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)
        self.output_layer = self.compiled_model.output(0)

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict]:
        input_tensor = self.preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        return self.postprocess(output, frame.shape, conf_threshold)

    def postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"
    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = 0.1  # Lowered for debugging
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        # Debug: print raw output shape
        print(f"[DEBUG] Model output shape: {output.shape}")
        detections = self._postprocess(output, frame.shape, conf_threshold)
        print(f"[DEBUG] Detections after postprocess: {len(detections)}")
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        print(f"[DEBUG] Raw detections before NMS: {len(detections)}")
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        print(f"[DEBUG] Detections after NMS: {len(detections)}")
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = 0.1  # Lowered for debugging
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        # Debug: print raw output shape
        print(f"[DEBUG] Model output shape: {output.shape}")
        detections = self._postprocess(output, frame.shape, conf_threshold)
        print(f"[DEBUG] Detections after postprocess: {len(detections)}")
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        print(f"[DEBUG] Raw detections before NMS: {len(detections)}")
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        print(f"[DEBUG] Detections after NMS: {len(detections)}")
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

# --- Video/Image/Live Inference ---
def run_inference(detector: OpenVINOYOLODetector, source=0, conf_threshold=0.25, flip=False, use_popup=False, video_width=None):
    if isinstance(source, str) and not os.path.exists(source):
        print(f"Downloading sample video: {source}")
        import requests
        url = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"
        r = requests.get(url)
        with open(source, 'wb') as f:
            f.write(r.content)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
    window_name = "YOLOv11x + OpenVINO Detection"
    if use_popup:
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    frame_count = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        if video_width:
            scale = video_width / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start = time.time()
        detections = detector.infer(frame, conf_threshold=conf_threshold)
        frame = detector.draw(frame, detections)
        elapsed = time.time() - start
        times.append(elapsed)
        if len(times) > 200:
            times.pop(0)
        fps = 1.0 / np.mean(times) if times else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if use_popup:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

# --- Main Entrypoint ---
if __name__ == "__main__":
    # Choose model: yolo11x or yolo11n, etc.
    MODEL_NAME = "yolo11x"

    DEVICE = "AUTO"  # or "CPU", "GPU"
    # Step 1: Convert model if needed
    ov_xml = convert_yolo_to_openvino(MODEL_NAME)
    # Step 2: Quantize (optional, demo skips actual quantization)
    ov_xml = quantize_openvino_model(ov_xml, MODEL_NAME)
    # Step 3: Create detector
    detector = OpenVINOYOLODetector(ov_xml, device=DEVICE)
    # Step 4: Run on webcam, video, or image
    # Webcam: source=0, Video: source="video.mp4", Image: source="image.jpg"
    run_inference(detector, source=0, conf_threshold=0.25, flip=True, use_popup=True, video_width=1280)
# To run on a video file: run_inference(detector, source="people.mp4", conf_threshold=0.25)
# To run on an image: run_inference(detector, source="image.jpg", conf_threshold=0.25)
# To run async or batch, extend the OpenVINOYOLODetector class with async API as needed.

import numpy as np
import cv2

def postprocess_openvino_yolo(output, conf_threshold=0.4, iou_threshold=0.5, input_shape=(640, 640), original_shape=None):
    """
    output: OpenVINO raw output tensor (e.g., shape [1, 25200, 85])
    conf_threshold: minimum confidence
    iou_threshold: for NMS
    input_shape: model input size (w, h)
    original_shape: original image size (w, h)
    """
    # 1. Squeeze batch dimension
    output = np.squeeze(output)  # [25200, 85]
    
    # 2. Split predictions
    boxes = output[:, :4]
    obj_conf = output[:, 4]
    class_scores = output[:, 5:]
    
    # 3. Get class with highest score
    class_ids = np.argmax(class_scores, axis=1)
    class_conf = class_scores[np.arange(len(class_scores)), class_ids]
    
    # 4. Multiply objectness confidence with class confidence
    scores = obj_conf * class_conf

    # 5. Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if original_shape is not None:
        # Rescale boxes from input_shape to original image shape
        input_w, input_h = input_shape
        orig_w, orig_h = original_shape
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2

    # 6. Convert boxes to [x, y, w, h] format for OpenCV NMS
    boxes_xywh = []
    for box in boxes:
        x1, y1, x2, y2 = box
        boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])

    # 7. Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)

    # 8. Return filtered boxes
    result_boxes = []
    result_scores = []
    result_classes = []
    if len(boxes) > 0 and len(scores) > 0:
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for i in indices:
                i = int(i)
                result_boxes.append(boxes[i])
                result_scores.append(scores[i])
                result_classes.append(class_ids[i])
    return result_boxes, result_scores, result_classes

import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional

# Only traffic-related classes for detection
TRAFFIC_CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
    'traffic light', 'stop sign', 'parking meter'
]

class OpenVINOVehicleDetector:
    def __init__(self, model_path: str = None, device: str = "AUTO", use_quantized: bool = False, enable_ocr: bool = False, confidence_threshold: float = 0.4):
        import openvino as ov
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ocr_reader = None
        self.class_names = TRAFFIC_CLASS_NAMES
        self.performance_stats = {
            'fps': 0,
            'avg_inference_time': 0,
            'frames_processed': 0,
            'backend': f"OpenVINO-{device}",
            'total_detections': 0,
            'detection_rate': 0
        }
        self._inference_times = []
        self._start_time = time.time()
        self._frame_count = 0
        # Model selection logic
        self.model_path = self._find_best_model(model_path, use_quantized)
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)
        # Always reshape to static shape before accessing .shape
        self.model.reshape({0: [1, 3, 640, 640]})
        self.input_shape = self.model.inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.ov_config = {}
        if device != "CPU":
            # Already reshaped above, so nothing more needed here
            pass
        if "GPU" in device or ("AUTO" in device and "GPU" in self.core.available_devices):
            self.ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device, config=self.ov_config)

        self.output_layer = self.compiled_model.output(0)

    def _find_best_model(self, model_path, use_quantized):
        # Priority: quantized IR > IR > .pt
        search_paths = [
            Path(model_path) if model_path else None,
            Path("yolo11x_openvino_int8_model/yolo11x.xml") if use_quantized else None,
            Path("yolo11x_openvino_model/yolo11x.xml"),
            Path("rcb/yolo11x_openvino_model/yolo11x.xml"),
            Path("yolo11x.xml"),
            Path("rcb/yolo11x.xml"),
            Path("yolo11x.pt"),
            Path("rcb/yolo11x.pt")
        ]
        for p in search_paths:
            if p and p.exists():
                return str(p)
        raise FileNotFoundError("No suitable YOLOv11x model found for OpenVINO.")

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict]:
        if conf_threshold is None:
            conf_threshold = 0.1  # Lowered for debugging
        start = time.time()
        input_tensor = self._preprocess(frame)
        output = self.compiled_model([input_tensor])[self.output_layer]
        # Debug: print raw output shape
        print(f"[DEBUG] Model output shape: {output.shape}")
        detections = self._postprocess(output, frame.shape, conf_threshold)
        print(f"[DEBUG] Detections after postprocess: {len(detections)}")
        elapsed = time.time() - start
        self._inference_times.append(elapsed)
        self._frame_count += 1
        self.performance_stats['frames_processed'] = self._frame_count
        self.performance_stats['total_detections'] += len(detections)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)
        self.performance_stats['avg_inference_time'] = float(np.mean(self._inference_times)) if self._inference_times else 0
        total_time = time.time() - self._start_time
        self.performance_stats['fps'] = self._frame_count / total_time if total_time > 0 else 0
        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]
        return img

    def _postprocess(self, output: np.ndarray, frame_shape, conf_threshold: float) -> List[Dict]:
        # Output: (1, 84, 8400) or (84, 8400) or (8400, 84)
        if output.ndim == 3:
            output = np.squeeze(output)
        if output.shape[0] == 84:
            output = output.T  # (8400, 84)
        boxes = output[:, :4]
        scores = output[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        detections = []
        h, w = frame_shape[:2]
        for i, (box, score, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if score < conf_threshold:
                continue
            x_c, y_c, bw, bh = box
            # If normalized, scale to input size
            if all(0.0 <= v <= 1.0 for v in box):
                x_c *= self.input_width
                y_c *= self.input_height
                bw *= self.input_width
                bh *= self.input_height
            # Scale to original frame size
            scale_x = w / self.input_width
            scale_y = h / self.input_height
            x_c *= scale_x
            y_c *= scale_y
            bw *= scale_x
            bh *= scale_y
            x1 = int(round(x_c - bw / 2))
            y1 = int(round(y_c - bh / 2))
            x2 = int(round(x_c + bw / 2))
            y2 = int(round(y_c + bh / 2))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only keep class 9 as traffic light, rename if found
            if class_id == 9:
                class_name = "traffic light"
            elif class_id < len(TRAFFIC_CLASS_NAMES):
                class_name = TRAFFIC_CLASS_NAMES[class_id]
            else:
                continue  # Remove unknown/other classes
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(score),
                'class_id': int(class_id),
                'class_name': class_name
            })
        print(f"[DEBUG] Raw detections before NMS: {len(detections)}")
        # Apply NMS
        if len(detections) > 0:
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, 0.5)
            if isinstance(indices, (list, tuple)) and len(indices) > 0:
                indices = np.array(indices).flatten()
            elif isinstance(indices, np.ndarray) and indices.size > 0:
                indices = indices.flatten()
            else:
                indices = []
            detections = [detections[int(i)] for i in indices] if len(indices) > 0 else []
        print(f"[DEBUG] Detections after NMS: {len(detections)}")
        return detections

    def draw(self, frame: np.ndarray, detections: List[Dict], box_thickness: int = 2) -> np.ndarray:
        # 80+ visually distinct colors for COCO classes (BGR)
        COCO_COLORS = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49),
            (72, 249, 10), (146, 204, 23), (61, 219, 134), (26, 147, 52), (0, 212, 187),
            (44, 153, 168), (0, 194, 255), (52, 69, 147), (100, 115, 255), (0, 24, 236),
            (132, 56, 255), (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
            (255, 255, 56), (255, 255, 151), (255, 255, 31), (255, 255, 29), (207, 255, 49),
            (72, 255, 10), (146, 255, 23), (61, 255, 134), (26, 255, 52), (0, 255, 187),
            (44, 255, 168), (0, 255, 255), (52, 255, 147), (100, 255, 255), (0, 255, 236),
            (132, 255, 255), (82, 255, 133), (203, 255, 255), (255, 255, 200), (255, 255, 199),
            (56, 255, 255), (157, 255, 151), (112, 255, 31), (178, 255, 29), (210, 255, 49),
            (249, 255, 10), (204, 255, 23), (219, 255, 134), (147, 255, 52), (212, 255, 187),
            (153, 255, 168), (194, 255, 255), (69, 255, 147), (115, 255, 255), (24, 255, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49),
            (249, 72, 10), (204, 146, 23), (219, 61, 134), (147, 26, 52), (212, 0, 187),
            (153, 44, 168), (194, 0, 255), (69, 52, 147), (115, 100, 255), (24, 0, 236),
            (56, 132, 255), (157, 82, 151), (112, 203, 31), (178, 255, 29), (210, 255, 49)
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = COCO_COLORS[det['class_id'] % len(COCO_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame