"""
Final Video Controller for Automatic Traffic Red-Light Violation Detection
- Uses detection_openvino.py for OpenVINO YOLOv11n detection
- Crosswalk (zebra crossing) detection using RANSAC/white-line logic
- Vehicle tracking using OpenCV trackers
- Violation logic: detects vehicles crossing the violation line on red
- Visualization and video output
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import cv2
import numpy as np
from sklearn import linear_model


# --- Crosswalk (Zebra Crossing) Detection ---
def detect_crosswalk(frame):
    """Detect crosswalk (zebra crossing) in the frame. Returns dict with detection status and y position."""
    # White color mask
    lower = np.array([170, 170, 170])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lower, upper)
    # Erode to remove noise
    erode_size = max(1, frame.shape[0] // 30)
    erode_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, 1))
    eroded = cv2.erode(mask, erode_structure, (-1, -1))
    # Find contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    left_points, right_points = [], []
    bw_width = 170
    crosswalk_y = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > bw_width:
            left_points.append([x, y])
            right_points.append([x + w, y])
    # RANSAC fit
    crosswalk_detected = False
    if len(left_points) > 5 and len(right_points) > 5:
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        model_l = linear_model.RANSACRegressor().fit(left_points[:, 0:1], left_points[:, 1])
        model_r = linear_model.RANSACRegressor().fit(right_points[:, 0:1], right_points[:, 1])
        # If the lines are roughly parallel and horizontal, assume crosswalk
        slope_l = model_l.estimator_.coef_[0]
        slope_r = model_r.estimator_.coef_[0]
        if abs(slope_l) < 0.3 and abs(slope_r) < 0.3:
            crosswalk_detected = True
            crosswalk_y = int(np.median(left_points[:, 1]))
    return {'crosswalk_detected': crosswalk_detected, 'crosswalk_y': crosswalk_y}

def get_traffic_light_color(frame, bbox):
    """Detect traffic light color in the given bounding box (x_min, y_min, x_max, y_max). Returns 'red', 'yellow', 'green', or 'unknown'."""
    x_min, y_min, x_max, y_max = bbox
    roi = frame[max(0, y_min):y_max, max(0, x_min):x_max]
    if roi.size == 0:
        return 'unknown'
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_red2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, (15, 70, 50), (35, 255, 255))
    mask_green = cv2.inRange(hsv, (40, 70, 50), (90, 255, 255))
    red = np.sum(mask_red)
    yellow = np.sum(mask_yellow)
    green = np.sum(mask_green)
    if max(red, yellow, green) == 0:
        return 'unknown'
    if red >= yellow and red >= green:
        return 'red'
    elif yellow >= green:
        return 'yellow'
    else:
        return 'green'
    
    ##model manager working
    import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# Import OpenVINO modules
from detection_openvino import OpenVINOVehicleDetector
from red_light_violation_pipeline import RedLightViolationPipeline

# Import from our utils package
from utils.helpers import bbox_iou

class ModelManager:
    """
    Manages OpenVINO models for traffic detection and violation monitoring.
    Only uses RedLightViolationPipeline for all violation/crosswalk/traffic light logic.
    """
    def __init__(self, config_file: str = None):
        """
        Initialize model manager with configuration.
        
        Args:
            config_file: Path to JSON configuration file
        """
        self.config = self._load_config(config_file)
        self.detector = None
        self.violation_pipeline = None  # Use RedLightViolationPipeline only
        self.tracker = None
        self._initialize_models()
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            Configuration dictionary
        """
        import json
        default_config = {
            "detection": {
                "confidence_threshold": 0.5,
                "enable_ocr": True,
                "enable_tracking": True,
                "model_path": None
            },
            "violations": {
                "red_light_grace_period": 2.0,
                "stop_sign_duration": 2.0,
                "speed_tolerance": 5
            },
            "display": {
                "max_display_width": 800,
                "show_confidence": True,
                "show_labels": True,
                "show_license_plates": True
            },
            "performance": {
                "max_history_frames": 1000,
                "cleanup_interval": 3600
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults (preserving loaded values)
                    for section in default_config:
                        if section in loaded_config:
                            default_config[section].update(loaded_config[section])
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def _initialize_models(self):
        """Initialize OpenVINO detection and violation models."""
        try:
            # Find best model path
            model_path = self.config["detection"].get("model_path")
            if not model_path or not os.path.exists(model_path):                
                model_path = self._find_best_model_path()
                if not model_path:
                    print("❌ No model found")
                    return
                
            # Initialize detector
            print(f"✅ Initializing OpenVINO detector with model: {model_path}")
            device = self.config["detection"].get("device", "AUTO")
            print(f"✅ Using inference device: {device}")
            self.detector = OpenVINOVehicleDetector(
                model_path=model_path,
                device=device,
                confidence_threshold=self.config["detection"]["confidence_threshold"]
            )
            
            # Use only RedLightViolationPipeline for violation/crosswalk/traffic light logic
            self.violation_pipeline = RedLightViolationPipeline(debug=True)
            print("✅ Red light violation pipeline initialized (all other violation logic removed)")
            
            # Initialize tracker if enabled
            if self.config["detection"]["enable_tracking"]:
                try:
                    from deep_sort_realtime.deepsort_tracker import DeepSort
                    
                    # Use optimized OpenVINO embedder if available
                    use_optimized_embedder = True
                    embedder = None
                    
                    if use_optimized_embedder:
                        try:
                            # Try importing our custom OpenVINO embedder
                            from utils.embedder_openvino import OpenVINOEmbedder
                            print(f"✅ Initializing optimized OpenVINO embedder on {device}")
                            
                            # Set model_path explicitly to use the user-supplied model
                            script_dir = Path(__file__).parent.parent
                            model_file_path = None
                            
                            # Try the copy version first (might be modified for compatibility)
                            copy_model_path = script_dir / "mobilenetv2 copy.xml"
                            original_model_path = script_dir / "mobilenetv2.xml"
                            
                            if copy_model_path.exists():
                                model_file_path = str(copy_model_path)
                                print(f"✅ Using user-supplied model: {model_file_path}")
                            elif original_model_path.exists():
                                model_file_path = str(original_model_path)
                                print(f"✅ Using user-supplied model: {model_file_path}")
                            
                            embedder = OpenVINOEmbedder(
                                model_path=model_file_path,
                                device=device,
                                half=True  # Use FP16 for better performance
                            )
                        except Exception as emb_err:
                            print(f"⚠️ OpenVINO embedder failed: {emb_err}, falling back to default")
                    
                    # Initialize tracker with embedder based on available parameters
                    if embedder is None:
                        print("⚠️ No embedder available, using DeepSORT with default tracking")
                    else:
                        print("✅ Initializing DeepSORT with custom embedder")
                    
                    # Simple initialization without problematic parameters
                    self.tracker = DeepSort(
                        max_age=30,
                        n_init=3,
                        nn_budget=100,
                        embedder=embedder
                    )
                    print("✅ DeepSORT tracker initialized")
                except ImportError:                    
                    print("⚠️ DeepSORT not available")
                    self.tracker = None
            print("✅ Models initialized successfully")
        
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            import traceback
            traceback.print_exc()
            
    def _find_best_model_path(self, base_model_name: str = None) -> Optional[str]:
        """
        Find best available model file in workspace.
        
        Args:
            base_model_name: Base model name without extension
            
        Returns:
            Path to model file or None
        """
        # Select model based on device if base_model_name is not specified
        if base_model_name is None:
            device = self.config["detection"].get("device", "AUTO")
            if device == "CPU" or device == "AUTO":
                # Use yolo11n for CPU - faster, lighter model
                base_model_name = "yolo11n"
                print(f"🔍 Device is {device}, selecting {base_model_name} model (optimized for CPU)")
            else:
                # Use yolo11x for GPU - larger model with better accuracy
                base_model_name = "yolo11x"
                print(f"🔍 Device is {device}, selecting {base_model_name} model (optimized for GPU)")
        
        # Check if the openvino_models directory exists in the current working directory
        cwd_openvino_dir = Path.cwd() / "openvino_models"
        if cwd_openvino_dir.exists():
            direct_path = cwd_openvino_dir / f"{base_model_name}.xml"
            if direct_path.exists():
                print(f"✅ Found model directly in CWD: {direct_path}")
                return str(direct_path.absolute())
                
        # Check for absolute path to openvino_models (this is the most reliable)
        absolute_openvino_dir = Path("D:/Downloads/finale6/khatam/openvino_models")
        if absolute_openvino_dir.exists():
            direct_path = absolute_openvino_dir / f"{base_model_name}.xml"
            if direct_path.exists():
                print(f"✅ Found model at absolute path: {direct_path}")
                return str(direct_path.absolute())
        
        # Try relative to the model_manager.py file
        openvino_models_dir = Path(__file__).parent.parent.parent / "openvino_models"
        direct_path = openvino_models_dir / f"{base_model_name}.xml"
        if direct_path.exists():
            print(f"✅ Found model in app directory: {direct_path}")
            return str(direct_path.absolute())
            
        # Check for model in folder structure within openvino_models
        subfolder_path = openvino_models_dir / f"{base_model_name}_openvino_model" / f"{base_model_name}.xml"
        if subfolder_path.exists():
            print(f"✅ Found model in subfolder: {subfolder_path}")
            return str(subfolder_path.absolute())
            
        # Try other common locations
        search_dirs = [
            ".",
            "..",
            "../models",
            "../rcb",
            "../openvino_models",
            f"../{base_model_name}_openvino_model",
            "../..",  # Go up to project root
            "../../openvino_models",  # Project root / openvino_models
        ]
        
        model_extensions = [
            (f"{base_model_name}.xml", "OpenVINO IR direct"),
            (f"{base_model_name}_openvino_model/{base_model_name}.xml", "OpenVINO IR"),
            (f"{base_model_name}.pt", "PyTorch"),
        ]
        
        for search_dir in search_dirs:
            search_path = Path(__file__).parent.parent / search_dir
            if not search_path.exists():
                continue
                
            for model_file, model_type in model_extensions:
                model_path = search_path / model_file
                if model_path.exists():
                    print(f"✅ Found {model_type} model: {model_path}")
                    return str(model_path.absolute())
        
        print(f"❌ No model found for {base_model_name}")
        return None
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of detection dictionaries
        """
        if self.detector is None:
            print("WARNING: No detector available")
            return []
        try:
            # Use a lower confidence threshold for better visibility
            conf_threshold = max(0.3, self.config["detection"].get("confidence_threshold", 0.5))
            detections = self.detector.detect_vehicles(frame, conf_threshold=conf_threshold)
            
            # Add debug output
            if detections:
                print(f"DEBUG: Detected {len(detections)} objects: " + 
                     ", ".join([f"{d['class_name']} ({d['confidence']:.2f})" for d in detections[:3]]))
                
                # Print bounding box coordinates of first detection
                if len(detections) > 0:
                    print(f"DEBUG: First detection bbox: {detections[0]['bbox']}")
            else:
                print("DEBUG: No detections in this frame")
                
            return detections
        except Exception as e:
            print(f"❌ Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def update_tracking(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracking information for detections.
        
        Args:
            detections: List of detections
            frame: Current video frame
            
        Returns:
            Updated list of detections with tracking info
        """
        if not self.tracker or not detections:
            return detections
        
        try:
            # Format detections for DeepSORT
            tracker_dets = []
            for det in detections:
                if 'bbox' not in det:
                    continue
                
                bbox = det['bbox']
                if len(bbox) < 4:
                    continue
                
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                
                if w <= 0 or h <= 0:
                    continue
                
                conf = det.get('confidence', 0.0)
                class_name = det.get('class_name', 'unknown')
                tracker_dets.append(([x1, y1, w, h], conf, class_name))
            
            # Update tracks
            if tracker_dets:
                tracks = self.tracker.update_tracks(tracker_dets, frame=frame)
                
                # Associate tracks with detections
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                        
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    
                    for det in detections:
                        if 'bbox' not in det:
                            continue
                            
                        bbox = det['bbox']
                        if len(bbox) < 4:
                            continue
                            
                        dx1, dy1, dx2, dy2 = bbox
                        iou = bbox_iou((dx1, dy1, dx2, dy2), tuple(map(int, ltrb)))
                        
                        if iou > 0.5:
                            det['track_id'] = track_id
                            break            
            return detections
            
        except Exception as e:
            print(f"❌ Tracking error: {e}")
            return detections
            
    def update_config(self, new_config: Dict):
        """
        Update configuration parameters.
        
        Args:
            new_config: New configuration dictionary
        """
        if not new_config:
            return
        
        # Store old device setting to check if it changed
        old_device = self.config["detection"].get("device", "AUTO") if "detection" in self.config else "AUTO"
            
        # Update configuration
        for section in new_config:
            if section in self.config:
                self.config[section].update(new_config[section])
            else:
                self.config[section] = new_config[section]
        
        # Check if device changed - if so, we need to reinitialize models
        new_device = self.config["detection"].get("device", "AUTO")
        device_changed = old_device != new_device
        
        if device_changed:
            print(f"📢 Device changed from {old_device} to {new_device}, reinitializing models...")
            # Reinitialize models with new device
            self._initialize_models()
            return
            
        # Just update detector confidence threshold if device didn't change
        if self.detector:
            conf_thres = self.config["detection"].get("confidence_threshold", 0.5)
            self.detector.conf_thres = conf_thres
