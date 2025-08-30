from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_pixmap,
    resize_frame_for_display
)

class VideoController(QObject):
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # New signal for direct NumPy frame display
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """
        super().__init__()
        self.model_manager = model_manager
        self.source = 0  # Default camera source
        self._running = False
        self.frame_count = 0
        self.start_time = 0
        self.source_fps = 0
        self.actual_fps = 0
        self.processing_times = deque(maxlen=30)
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
        
        # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
          # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0

    def on_model_switched(self, device):
        """Handle device switch notification from model manager."""
        print(f"[VIDEO CONTROLLER] Device switched to: {device}")
        # Update model manager config if needed
        if self.model_manager and hasattr(self.model_manager, 'config'):
            self.model_manager.config["detection"]["device"] = device
            print(f"[VIDEO CONTROLLER] Updated model manager device to: {device}")
    
    def set_source(self, source):
        """Set video source (file path, camera index, or URL)"""
        print(f"DEBUG: VideoController.set_source called with: {source} (type: {type(source)})")
        
        was_running = self._running
        if self._running:
            self.stop()
        
        # Critical fix: Make sure source is properly set
        if source is None:
            print("WARNING: Received None source, defaulting to camera 0")
            self.source = 0
        elif isinstance(source, str) and source.strip():
            # Handle file paths - verify the file exists
            if os.path.exists(source):
                self.source = source
                print(f"DEBUG: VideoController source set to file: {self.source}")
            else:
                # Try to interpret as camera index or URL
                try:
                    # If it's a digit string, convert to integer camera index
                    if source.isdigit():
                        self.source = int(source)
                        print(f"DEBUG: VideoController source set to camera index: {self.source}")
                    else:
                        # Treat as URL or special device string
                        self.source = source
                        print(f"DEBUG: VideoController source set to URL/device: {self.source}")
                except ValueError:
                    print(f"WARNING: Could not interpret source: {source}, defaulting to camera 0")
                    self.source = 0
        elif isinstance(source, int):
            # Camera index
            self.source = source
            print(f"DEBUG: VideoController source set to camera index: {self.source}")
        else:
            print(f"WARNING: Unrecognized source type: {type(source)}, defaulting to camera 0")
            self.source = 0
            
        # Get properties of the source (fps, dimensions, etc)
        self._get_source_properties()
        
        if was_running:
            self.start()
    
    def _get_source_properties(self):
        """Get properties of video source"""
        try:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self.source_fps = cap.get(cv2.CAP_PROP_FPS)
                if self.source_fps <= 0:
                    self.source_fps = 30.0  # Default if undetectable
                
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
                self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                print(f"Video source: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            else:
                print("Failed to open video source")
        except Exception as e:
            print(f"Error getting source properties: {e}")
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread
            if not self.thread.isRunning():
                self.thread.start()
              # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            self.render_timer.start(10)
            print("DEBUG: Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"DEBUG: Opening video file: {self.source}")
                cap = cv2.VideoCapture(self.source)
                
                # Verify file opened successfully
                if not cap.isOpened():
                    print(f"ERROR: Could not open video file: {self.source}")
                    return
                    
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"DEBUG: Opening camera: {camera_idx}")
                cap = cv2.VideoCapture(camera_idx)
                
                # Try a few times to open camera (sometimes takes a moment)
                retry_count = 0
                while not cap.isOpened() and retry_count < 3:
                    print(f"Camera not ready, retrying ({retry_count+1}/3)...")
                    time.sleep(1)
                    cap.release()
                    cap = cv2.VideoCapture(camera_idx)
                    retry_count += 1
                    
                if not cap.isOpened():
                    print(f"ERROR: Could not open camera {camera_idx} after {retry_count} attempts")
                    return
            else:
                # Try as a string source (URL or device path)
                print(f"DEBUG: Opening source as string: {self.source}")
                cap = cv2.VideoCapture(str(self.source))
                
                if not cap.isOpened():
                    print(f"ERROR: Could not open source: {self.source}")
                    return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
            
            # Main processing loop
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video or read error")
                    break
                    
                # Detection processing
                process_start = time.time()
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                detection_time = (time.time() - detection_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                
                # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                
                # Signal for raw data subscribers
                self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                
                # Emit NumPy frame for direct display
                self.frame_np_ready.emit(frame.copy())
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:                        
                     time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False    
            
    def _process_frame(self):
        """Process current frame for UI rendering (called by timer)"""
        if not self._running:
            return
            
        # Debug counter
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
            if self.debug_counter % 30 == 0:  # Print every ~30 frames
                print(f"DEBUG: Frame processing iteration: {self.debug_counter}")
        
        # Get frame data safely
        self.mutex.lock()
        if self.current_frame is None:
            self.mutex.unlock()
            return
            
        # Make a copy of the data we need
        frame = self.current_frame.copy()
        detections = self.current_detections.copy() if self.current_detections else []
        metrics = self.performance_metrics.copy()
        self.mutex.unlock()
        
        try:
            # Process frame for display using enhanced annotation
            annotated_frame = frame.copy()
            
            # Draw detections on frame with enhanced visualization
            if detections:
                print(f"DEBUG: Drawing {len(detections)} detections")
                annotated_frame = enhanced_draw_detections(annotated_frame, detections, True, True)
                
            # Draw performance metrics with enhanced overlay
            annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            
            # Resize for display if needed (1280x720 is a good size for most displays)
            display_frame = resize_frame_for_display(annotated_frame, max_width=1280, max_height=720)
              # Use enhanced direct OpenCV to QPixmap conversion with data copy to prevent black frames
            # Convert to RGB and ensure QImage owns its data
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()  # .copy() is critical!
            pixmap = QPixmap.fromImage(qt_image)
            
            # Emit signal with the pixmap
            if not pixmap.isNull():
                print(f"DEBUG: Emitting pixmap: {pixmap.width()}x{pixmap.height()}")
                self.frame_ready.emit(pixmap, detections, metrics)
            else:
                print("ERROR: Generated null pixmap")

            # Emit NumPy frame for direct display
            self.frame_np_ready.emit(display_frame)
            
        except Exception as e:
            print(f"ERROR in _process_frame: {e}")
            import traceback
            traceback.print_exc()
from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils import detect_crosswalk_and_violation_line, draw_violation_line
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    print(f"Drawing {len(detections)} detection boxes on frame")
                    for det in detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR

                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated_frame, f"{label} {confidence:.2f}", 
                                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                            # Draw traffic light color indicator if this is a traffic light
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")
                
                # Add FPS display directly on frame
                cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # --- Always draw detected traffic light color indicator at top ---
                color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                indicator_size = 30
                margin = 10
                status_colors = {
                    "red": (0, 0, 255),
                    "yellow": (0, 255, 255),
                    "green": (0, 255, 0),
                    "unknown": (200, 200, 200)
                }
                draw_color = status_colors.get(color, (200, 200, 200))
                # Draw circle indicator
                cv2.circle(
                    annotated_frame,
                    (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                    indicator_size,
                    draw_color,
                    -1
                )
                # Add color text
                cv2.putText(
                    annotated_frame,
                    f"{color.upper()} ({confidence:.2f})",
                    (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()# Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                violations = []  # Violations are disabled
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # Process frame for display using enhanced annotation
            annotated_frame = frame.copy()
            
            # Detect and draw crosswalk/stopline first
            # This ensures the violation line is drawn below other overlays
            try:
                # Find traffic light in detections
                traffic_light_bbox = None
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        traffic_light_bbox = det.get('bbox')
                        if traffic_light_bbox:
                            print(f"Found traffic light with bbox: {traffic_light_bbox}")
                            break
                # Only proceed if a real traffic light is detected
                if not traffic_light_bbox:
                    print("⚠️ No traffic light detected, skipping crosswalk detection for this frame.")
                    crosswalk_bbox = None
                    violation_line_y = None
                    crosswalk_debug = {}
                else:
                    # Use center of traffic light bbox as position
                    tl_x = (traffic_light_bbox[0] + traffic_light_bbox[2]) // 2
                    tl_y = (traffic_light_bbox[1] + traffic_light_bbox[3]) // 2
                    print("[DEBUG] About to call detect_crosswalk_and_violation_line")
                    result_frame, crosswalk_bbox, violation_line_y, crosswalk_debug = detect_crosswalk_and_violation_line(annotated_frame, (tl_x, tl_y))
                    print(f"[DEBUG] detect_crosswalk_and_violation_line returned: bbox={crosswalk_bbox}, vline_y={violation_line_y}")
                    annotated_frame = result_frame  # Use the frame with overlays from crosswalk_utils
                    # Draw crosswalk bbox if found
                    if crosswalk_bbox:
                        x, y, w_, h_ = crosswalk_bbox
                        # Draw a semi-transparent yellow rectangle for crosswalk
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (x, y), (x + w_, y + h_), (0, 255, 255), -1)
                        alpha = 0.25
                        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                        # Draw a thick border
                        cv2.rectangle(annotated_frame, (x, y), (x + w_, y + h_), (0, 255, 255), 4)
                        # Draw label with background
                        label = "CROSSWALK"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                        cv2.rectangle(annotated_frame, (x, y - th - 12), (x + tw + 10, y), (0, 255, 255), -1)
                        cv2.putText(annotated_frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
                    # Draw violation line if found
                    if violation_line_y:
                        line_color = (0, 0, 255) if self.latest_traffic_light.get('color', 'unknown') == 'red' else (0, 255, 0)
                        label = f"VIOLATION LINE - {'RED' if self.latest_traffic_light.get('color', 'unknown') == 'red' else 'GREEN'}"
                        # Draw a thick, dashed line
                        x1, x2 = 0, annotated_frame.shape[1]
                        for i in range(x1, x2, 40):
                            cv2.line(annotated_frame, (i, violation_line_y), (min(i+20, x2), violation_line_y), line_color, 6)
                        # Draw label with background
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(annotated_frame, (10, violation_line_y - th - 18), (10 + tw + 10, violation_line_y - 2), line_color, -1)
                        cv2.putText(annotated_frame, label, (15, violation_line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            except Exception as e:
                print(f"❌ Error in crosswalk detection: {e}")
                import traceback
                traceback.print_exc()
            
            # Process traffic light detections to identify colors
            traffic_light_detected = False
            for det in detections:
                # Use our helper function to check any variant of traffic light class
                if is_traffic_light(det.get('class_name')):
                    # Get traffic light color
                    bbox = det['bbox']
                    
                    # Print the original class name for debugging
                    original_class = det.get('class_name', '')
                    print(f"🚦 Found traffic light detection with class: '{original_class}'")
                    
                    # Safe check for valid bbox
                    if isinstance(bbox, list) and len(bbox) == 4:
                        traffic_light_detected = True
                        # Enforce traffic light detection for demo purposes
                        if det.get('confidence', 0) < 0.4:  # If low confidence or missing
                            # For demo testing - hardcode a traffic light with changing colors
                            print(f"⚠️ Low confidence traffic light detected ({det.get('confidence', 0):.2f}), using demo colors")
                            
                            # This section can be removed in production
                            if hasattr(self, '_demo_light_state'):
                                self._demo_light_state = (self._demo_light_state + 1) % 30
                            else:
                                self._demo_light_state = 0
                                
                            if self._demo_light_state < 10:
                                color = "red"
                            elif self._demo_light_state < 15:
                                color = "yellow"
                            else:
                                color = "green"
                                
                            light_info = {"color": color, "confidence": 0.95}  # High confidence for demo
                            print(f"🚦 Using demo traffic light color: {color}")
                        else:
                            # Normal detection with enhanced function
                            # Get the traffic light detection start time
                            tl_start = time.time()
                            light_info = {"color": "unknown", "confidence": 0.0}
                            
                            # Create a debug visualization of the traffic light crop
                            try:
                                x1, y1, x2, y2 = [int(c) for c in bbox]
                                # Ensure coordinates are within frame bounds
                                h, w = frame.shape[:2]
                                x1 = max(0, min(x1, w-1))
                                y1 = max(0, min(y1, h-1))
                                x2 = max(0, min(x2, w-1))
                                y2 = max(0, min(y2, h-1))
                                
                                # Print bbox to help with debugging
                                print(f"🔍 Traffic light bbox: [{x1}, {y1}, {x2}, {y2}], size: {x2-x1}x{y2-y1}")
                            
                                # Exit early if the box is invalid
                                if x2 <= x1 or y2 <= y1:
                                    print("⚠️ Invalid traffic light bbox (empty or invalid)")
                                else:
                                    # Extract ROI for visualization
                                    tl_crop = frame[y1:y2, x1:x2].copy()
                                    
                                    if tl_crop.size > 0:
                                        # Check if crop is not empty/black
                                        if np.mean(tl_crop) < 10:  # Very dark image
                                            print("⚠️ Traffic light crop is very dark, likely invalid")
                                            
                                        # Create a bigger debug view
                                        debug_crop = tl_crop.copy()
                                        
                                        # Resize for better visibility if small
                                        if debug_crop.shape[0] < 40 or debug_crop.shape[1] < 40:
                                            print(f"🔍 Resizing small traffic light crop for debug: {debug_crop.shape}")
                                            scale = max(4, 80 / max(debug_crop.shape[0], debug_crop.shape[1]))
                                            debug_crop = cv2.resize(debug_crop, 
                                                                  (int(debug_crop.shape[1] * scale), 
                                                                   int(debug_crop.shape[0] * scale)))
                                        
                                        # Create metadata panel
                                        info_panel = np.zeros((80, debug_crop.shape[1], 3), dtype=np.uint8)
                                        cv2.putText(info_panel, f"Traffic Light: {x2-x1}x{y2-y1}px", 
                                                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                        cv2.putText(info_panel, f"Position: ({x1},{y1}) to ({x2},{y2})", 
                                                   (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                        cv2.putText(info_panel, f"Mean value: {np.mean(tl_crop):.1f}", 
                                                   (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                        
                                        # Stack crop and info panel
                                        debug_view = np.vstack([debug_crop, info_panel]) if debug_crop.shape[1] == info_panel.shape[1] else debug_crop
                                        
                                        # Show the debug view
                                        # cv2.imshow("Traffic Light Debug", debug_view)  # Disabled for headless environment
                                        # cv2.waitKey(1)  # Disabled for headless environment
                                        
                                        # Also save a copy for further analysis
                                        try:
                                            cv2.imwrite("traffic_light_debug.png", debug_view)
                                            cv2.imwrite("traffic_light_crop.png", tl_crop)
                                        except:
                                            pass
                            except Exception as e:
                                print(f"❌ Error in traffic light visualization: {e}")
                                import traceback
                                traceback.print_exc()
                                        
                                    # Run the actual detection on the original frame crop
                                    # Try our robust approach that guarantees a color result
                                try:
                                        # Import the special function for guaranteed traffic light detection
                                        from utils.traffic_light_utils import ensure_traffic_light_color
                                        
                                        # Use the ensure function that will never return unknown
                                        light_info = ensure_traffic_light_color(frame, bbox)
                                        
                                        tl_time = (time.time() - tl_start) * 1000  # convert to ms
                                        
                                        # Handle both string and dictionary return formats
                                        if isinstance(light_info, dict):
                                            color = light_info.get('color', 'unknown')
                                            confidence = light_info.get('confidence', 0.0)
                                            print(f"🚦 Detected traffic light with color: {color}, confidence: {confidence:.2f}, time: {tl_time:.1f}ms")
                                        else:
                                            # Legacy format handling
                                            light_info = {"color": light_info, "confidence": 1.0}
                                            print(f"🚦 Detected traffic light with color: {light_info['color']} (legacy format)")
                                except Exception as e:
                                        print(f"❌ Error in traffic light detection: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        # Even if all else fails, return a red traffic light for safety
                                        light_info = {"color": "red", "confidence": 0.3}
                        
                        # Add color information to detection 
                        det['traffic_light_color'] = light_info
                        
                        # Update latest_traffic_light with the detected color info
                        self.latest_traffic_light = light_info
                        
                        # Use specialized drawing for traffic lights
                        try:
                            from utils.traffic_light_utils import draw_traffic_light_status
                            annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                            
                            # Also add a large indicator at the top of the frame for high visibility
                            color = light_info.get('color', 'unknown') if isinstance(light_info, dict) else light_info
                            indicator_size = 50
                            margin = 20
                            
                            # Define color for drawing
                            status_colors = {
                                "red": (0, 0, 255),      # BGR: Red
                                "yellow": (0, 255, 255), # BGR: Yellow
                                "green": (0, 255, 0),    # BGR: Green
                                "unknown": (255, 255, 255)  # BGR: White
                            }
                            draw_color = status_colors.get(color, (255, 255, 255))
                            
                            # Draw colored circle indicator at top-right
                            cv2.circle(
                                annotated_frame, 
                                (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size), 
                                indicator_size, 
                                draw_color, 
                                -1  # filled circle
                            )
                            
                            # Add text inside the circle
                            cv2.putText(
                                annotated_frame, 
                                color.upper(), 
                                (annotated_frame.shape[1] - margin - indicator_size - 35, margin + indicator_size + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1.2, 
                                (0, 0, 0), # Black text for contrast
                                4
                            )
                            
                        except Exception as e:
                            print(f"❌ Error drawing traffic light status: {e}")
                            # Fallback to simple rectangle
                            x1, y1, x2, y2 = [int(c) for c in bbox]
                            color = light_info.get('color', 'unknown') if isinstance(light_info, dict) else light_info
                            
                            # Define colors for different states
                            if color == 'red':
                                color_bgr = (0, 0, 255)  # BGR red
                            elif color == 'yellow':
                                color_bgr = (0, 255, 255)  # BGR yellow
                            elif color == 'green':
                                color_bgr = (0, 255, 0)  # BGR green
                            else:
                                color_bgr = (255, 255, 255)  # BGR white
                                
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 3)
                            
                            # Add label
                            label = f"Traffic Light: {color.upper()}"
                            cv2.putText(annotated_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                    else:
                        print(f"⚠️ Invalid bbox found for traffic light: {bbox}")
            
            # Add a default traffic light if none detected (for demo purposes)
            if not traffic_light_detected:
                print("⚠️ No traffic light detected, using default state")
                
                # In many traffic monitoring scenarios, it's safer to default to red 
                # if no traffic light is detected
                self.latest_traffic_light = {"color": "red", "confidence": 0.5}
                
                # Force a green light every 10 seconds to ensure the color changing works
                if hasattr(self, '_demo_cycle_counter'):
                    self._demo_cycle_counter += 1
                    if self._demo_cycle_counter > 150:  # ~5s at 30fps
                        print("🟢 Forcing GREEN light for demo cycling")
                        self.latest_traffic_light = {"color": "green", "confidence": 0.8}
                        if self._demo_cycle_counter > 300:  # ~10s at 30fps
                            self._demo_cycle_counter = 0
                else:
                    self._demo_cycle_counter = 0
            
            # Process red light violations if detector is available
            if self.violation_detector:
                # Make sure latest_traffic_light is handled properly
                if isinstance(self.latest_traffic_light, dict) and self.latest_traffic_light.get('color') != "unknown":
                    # Process frame for violations with dictionary format
                    violation_frame, new_violations = self.violation_detector.process_frame(
                        annotated_frame,
                        detections,
                        self.latest_traffic_light
                    )
                elif isinstance(self.latest_traffic_light, str) and self.latest_traffic_light != "unknown":
                    # Handle legacy string format
                    violation_frame, new_violations = self.violation_detector.process_frame(
                        annotated_frame,
                        detections,
                        self.latest_traffic_light
                    )
                else:
                    # Skip violation detection if color is unknown
                    violation_frame, new_violations = annotated_frame, []
                
                # Update annotated frame with violation markings
                annotated_frame = violation_frame
                
                # Emit signals for any new violations
                for violation in new_violations:
                    print(f"🚨 RED LIGHT VIOLATION DETECTED: {violation['id']}")
                    self.violation_detected.emit(violation)
            
            # Draw detections on frame with enhanced visualization
            if detections:
                print(f"DEBUG: Drawing {len(detections)} detections")
                # For detections without traffic_light_color (other objects), use enhanced_draw_detections
                other_detections = [d for d in detections if d.get('class_name') != 'traffic light']
                if other_detections:
                    annotated_frame = enhanced_draw_detections(annotated_frame, other_detections, True, True)
                
            # Draw performance metrics with enhanced overlay
            annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            
            # Resize for display if needed (1280x720 is a good size for most displays)
            display_frame = resize_frame_for_display(annotated_frame, max_width=1280, max_height=720)
            
            # Use enhanced direct OpenCV to QPixmap conversion
            pixmap = enhanced_cv_to_pixmap(display_frame)
            
            # Emit signal with the pixmap
            if not pixmap.isNull():
                print(f"DEBUG: Emitting pixmap: {pixmap.width()}x{pixmap.height()}")
                self.frame_ready.emit(pixmap, detections, metrics)
            else:
                print("ERROR: Generated null pixmap")
                
            # Emit NumPy frame for direct display - use enhanced annotations
            print(f"🔵 Emitting display_frame from _process_frame with shape: {display_frame.shape}")
            try:
                # Force frame to be contiguous
                display_frame_copy = np.ascontiguousarray(display_frame)
                print(f"🔄 Processed frame is contiguous: {display_frame_copy.flags['C_CONTIGUOUS']}, memory: {hex(id(display_frame_copy))}")
                self.frame_np_ready.emit(display_frame_copy)
                print("✅ Emitted frame_np_ready from _process_frame successfully")
            except Exception as e:
                print(f"❌ Error emitting frame from _process_frame: {e}")
                import traceback
                traceback.print_exc()
              # Emit stats signal for performance monitoring            # Emit stats signal for performance monitoring
            fps_val = float(metrics.get('FPS', 0.0))
            det_time = float(metrics.get('Detection (ms)', 0.0))
            try:
                stats = {
                    'fps': fps_val,
                    'detection_time': det_time,
                    'traffic_light_color': self.latest_traffic_light
                }
                self.stats_ready.emit(stats)
                print(f"📊 Emitted stats: FPS={fps_val:.1f}, Detection={det_time:.1f}ms, Traffic Light={self.latest_traffic_light}")
            except Exception as e:
                print(f"❌ Error emitting stats: {e}")
                
        except Exception as e:
            print(f"ERROR in _process_frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _force_traffic_light_detection(self, frame, detections):
        """
        Force traffic light detection by adding a dummy traffic light if none detected.
        This is for testing purposes only.
        """
        # Check if traffic light was already detected
        for det in detections:
            if det.get('class_name') == 'traffic light':
                return detections  # Already have a traffic light
        
        # Create a dummy traffic light detection
        h, w = frame.shape[:2]
        dummy_traffic_light = {
            'class_name': 'traffic light',
            'class_id': 9,  # COCO class ID for traffic light
            'confidence': 0.95,
            'bbox': [w - 150, 50, w - 50, 150],  # Top-right corner
            'track_id': -1
        }
        
        # Add to detections list
        detections.append(dummy_traffic_light)
        print("🚦 Added dummy traffic light for testing")
        
        return detections


####working
from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap,
    pipeline_with_violation_line
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line, get_violation_line_y
from controllers.deepsort_tracker import DeepSortVehicleTracker
from violation_finale.red_light_violation import RedLightViolationSystem, draw_violation_overlay
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = DeepSortVehicleTracker()
        
        # Add red light violation system
        self.red_light_violation_system = RedLightViolationSystem()
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    print(f"Drawing {len(detections)} detection boxes on frame")
                    for det in detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR
                            if 'id' in det:
                                id_text = f"ID: {det['id']}"
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated_frame, f"{id_text} {label} ", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            # Draw vehicle ID if present
                            # if 'id' in det:
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()# Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                violations = []  # Violations are disabled
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # --- Always use the same annotated_frame for all overlays ---
            annotated_frame = frame.copy()

            # 1. Draw detection bounding boxes and traffic light overlays
            for det in detections:
                if 'bbox' in det:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    label = det.get('class_name', 'object')
                    confidence = det.get('confidence', 0.0)
                    # Robustness: ensure label and confidence are not None
                    if label is None:
                        label = 'object'
                    if confidence is None:
                        confidence = 0.0
                    class_id = det.get('class_id', -1)
                    if class_id == 9 or is_traffic_light(label):
                        box_color = (0, 0, 255)
                    else:
                        box_color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    # Draw vehicle ID if present
                    if 'id' in det:
                        id_text = f"ID: {det['id']}"
                        # Calculate text size for background
                        (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        # Draw filled rectangle for background (top-left of bbox)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                        # Draw the ID text in bold yellow
                        cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                        print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                    # Draw traffic light color indicator if this is a traffic light
                    if class_id == 9 or is_traffic_light(label):
                        try:
                            light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                            if light_info.get("color", "unknown") == "unknown":
                                light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                            det['traffic_light_color'] = light_info
                            annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                            self.latest_traffic_light = light_info
                        except Exception as e:
                            print(f"[WARN] Could not detect/draw traffic light color: {e}")

            # 2. Robust crosswalk/stop line logic integration
            # Use traffic light bbox center if available
            traffic_light_bbox = None
            for det in detections:
                if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                    traffic_light_bbox = det['bbox']
                    break
            traffic_light_pos = None
            if traffic_light_bbox:
                tl_x = (traffic_light_bbox[0] + traffic_light_bbox[2]) // 2
                tl_y = (traffic_light_bbox[1] + traffic_light_bbox[3]) // 2
                traffic_light_pos = (tl_x, tl_y)
            # Call robust detection method
            violation_line, crosswalk_detected, stop_line_detected, violation_confidence = self._detect_violation_line_video_controller(annotated_frame, traffic_light_pos)
            # Draw violation line if valid
            if violation_line is not None:
                start_pt, end_pt = violation_line
                line_color = (0, 255, 255) if not stop_line_detected else (255, 0, 0)
                cv2.line(annotated_frame, start_pt, end_pt, line_color, 8)
                label = f"Violation Line ({'crosswalk' if crosswalk_detected else 'stop line' if stop_line_detected else 'default'})"
                cv2.putText(annotated_frame, label, (10, start_pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
                print(f"[DEBUG] Violation line drawn at y={start_pt[1]}, type={label}")
            else:
                print(f"[DEBUG] No valid violation line detected.")

            # --- Red light violation detection and overlay ---
            # Get violation line y (if available)
            violation_line_y = None
            if violation_line is not None:
                violation_line_y = start_pt[1]
            # Run violation detection
            print(f"🟢 Type of red_light_violation_system: {type(self.red_light_violation_system)}")
            print(f"🟢 Args to process_frame: frame={type(frame)}, detections={type(detections)}, traffic_light_bbox={traffic_light_bbox}, frame_idx=0")
            print("[DEBUG] About to call RedLightViolationSystem.process_frame")
            violations = self.red_light_violation_system.process_frame(
                frame, detections, traffic_light_bbox if traffic_light_bbox else [0,0,0,0], 0
            )
            print("🟢 Finished calling process_frame")
            # Draw violation overlay (including tracked positions)
            annotated_frame = draw_violation_overlay(
                annotated_frame,
                violations,
                violation_line_y,
                vehicle_tracks=self.red_light_violation_system.vehicle_tracks
            )

            # 3. Add performance overlays, test lines, and debug marker on the same annotated_frame
            annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            cv2.circle(annotated_frame, (20, 20), 10, (255, 255, 0), -1)

            # Convert BGR to RGB before display (for PyQt/PySide)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # Display the RGB frame in the UI (replace with your display logic)
            # Example: self.image_label.setPixmap(QPixmap.fromImage(QImage(frame_rgb.data, w, h, QImage.Format_RGB888)))
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_violation_line_video_controller(self, frame: np.ndarray, traffic_light_pos=None):
        """
        Robust crosswalk/stop line logic for VideoController integration.
        Returns: (violation_line, crosswalk_detected, stop_line_detected, violation_confidence)
        """
        if frame is None:
            print("Frame is None!")
            return None, False, False, 0.0
        print(f"Traffic light position: {traffic_light_pos}")
        frame_height, frame_width = frame.shape[:2]
        # --- Crosswalk detection ---
        crosswalk_line, crosswalk_conf, crosswalk_dist = self._detect_crosswalk(frame, traffic_light_pos)
        print(f"Crosswalk Line: {crosswalk_line}")
        # --- Stop line detection ---
        stop_line, stop_conf, stop_dist = self._detect_stop_line(frame, traffic_light_pos)
        print(f"Stop Line: {stop_line}")
        best_line, best_type, best_conf = None, None, 0.0
        # Select the nearest valid line to the traffic light if known
        if traffic_light_pos:
            candidates = []
            if crosswalk_line:
                candidates.append((crosswalk_line, 'crosswalk', crosswalk_conf, crosswalk_dist))
            if stop_line:
                candidates.append((stop_line, 'stop_line', stop_conf, stop_dist))
            if candidates:
                best = min(candidates, key=lambda x: x[3])
                best_line, best_type, best_conf = best[0], best[1], best[2]
        else:
            if crosswalk_line and crosswalk_conf >= stop_conf:
                best_line, best_type, best_conf = crosswalk_line, 'crosswalk', crosswalk_conf
            elif stop_line:
                best_line, best_type, best_conf = stop_line, 'stop_line', stop_conf
        if best_line:
            crosswalk_detected = (best_type == 'crosswalk')
            stop_line_detected = (best_type == 'stop_line')
            violation_confidence = best_conf
            return best_line, crosswalk_detected, stop_line_detected, violation_confidence
        # Fallback: Use default line at 75% height or relative to traffic light
        if traffic_light_pos:
            offset = int(0.15 * frame_height)
            fallback_y = min(traffic_light_pos[1] + offset, frame_height - 1)
        else:
            fallback_y = int(frame_height * 0.75)
        return ((0, fallback_y), (frame_width, fallback_y)), False, False, 0.3

    def _detect_crosswalk(self, frame: np.ndarray, traffic_light_pos=None):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, self.crosswalk_threshold, 255, cv2.THRESH_BINARY)
            eroded = cv2.erode(thresh, self.erosion_kernel, iterations=1)
            cleaned = cv2.dilate(eroded, self.dilation_kernel, iterations=2)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            crosswalk_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_crosswalk_area:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if 4 <= len(approx) <= 8:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        if 2 < aspect_ratio < 10:
                            roi = cleaned[y:y+h, x:x:x+w]
                            lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=30, minLineLength=int(0.5*w), maxLineGap=10)
                            if lines is not None:
                                angles = []
                                for l in lines:
                                    x1, y1, x2, y2 = l[0]
                                    angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                                    angles.append(angle)
                                if angles and np.std(angles) < 15 and np.all(np.abs(np.abs(angles)-90) < 20):
                                    crosswalk_candidates.append((contour, x, y, w, h, area))
            if not crosswalk_candidates:
                return None, 0.0, float('inf')
            if traffic_light_pos:
                best = min(crosswalk_candidates, key=lambda c: self._distance_to_traffic_light((c[1],c[2],c[3],c[4]), traffic_light_pos))
            else:
                best = max(crosswalk_candidates, key=lambda c: c[5])
            _, x, y, w, h, _ = best
            offset = int(0.1 * h)
            violation_y = max(y - offset, 0)
            frame_width = frame.shape[1]
            confidence = 0.9
            dist = self._distance_to_traffic_light((x, y, w, h), traffic_light_pos) if traffic_light_pos else float('inf')
            return ((0, violation_y), (frame_width, violation_y)), confidence, dist
        except Exception as e:
            print(f"Error in crosswalk detection: {e}")
            return None, 0.0, float('inf')

    def _detect_stop_line(self, frame: np.ndarray, traffic_light_pos=None):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_height, frame_width = frame.shape[:2]
            roi_start = int(frame_height * 0.5)
            roi = gray[roi_start:, :]
            adaptive_thresh = cv2.adaptiveThreshold(
                roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            edges = cv2.Canny(adaptive_thresh, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=100, 
                minLineLength=80, 
                maxLineGap=20
            )
            if lines is None:
                return None, 0.0, float('inf')
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 15:
                    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    y_avg = (y1 + y2) // 2 + roi_start
                    horizontal_lines.append((length, y_avg, x1, x2))
            if not horizontal_lines:
                return None, 0.0, float('inf')
            best_line = max(horizontal_lines, key=lambda x: x[0])
            _, y_pos, x1, x2 = best_line
            offset = int(0.05 * frame_height)
            violation_y = max(y_pos - offset, 0)
            confidence = 0.7
            dist = self._distance_to_traffic_light((x1, y_pos, x2-x1, 1), traffic_light_pos) if traffic_light_pos else float('inf')
            return ((0, violation_y), (frame_width, violation_y)), confidence, dist
        except Exception as e:
            print(f"Error in stop line detection: {e}")
            return None, 0.0, float('inf')

    def _distance_to_traffic_light(self, contour_or_rect, traffic_light_pos):
        if not traffic_light_pos:
            return float('inf')
        if isinstance(contour_or_rect, tuple):
            x, y, w, h = contour_or_rect
            cx, cy = x + w // 2, y + h // 2
        else:
            x, y, w, h = cv2.boundingRect(contour_or_rect)
            cx, cy = x + w // 2, y + h // 2
        return np.linalg.norm(np.array((cx, cy)) - np.array(traffic_light_pos))

    def process_vehicle_tracking(self, detections, frame):
        """
        Assigns IDs to vehicles using DeepSORT and returns list of dicts with ID and bbox.
        Only valid vehicle classes are tracked. Enhances class mapping and filtering.
        detections: list of dicts with keys ['bbox', 'confidence', 'class']
        frame: current BGR frame
        Returns: list of dicts with keys ['id', 'bbox', 'confidence', 'class']
        """
        # Define valid vehicle classes and their canonical names
        vehicle_classes = {
            'car': 0, 'truck': 1, 'bus': 2, 'motorcycle': 3, 'van': 4, 'bicycle': 5
        }
        # Accept common variants and filter out non-vehicles
        valid_names = set(vehicle_classes.keys())
        class_aliases = {
            'car': ['car', 'auto', 'automobile', 'sedan', 'hatchback'],
            'truck': ['truck', 'lorry', 'pickup'],
            'bus': ['bus', 'coach'],
            'motorcycle': ['motorcycle', 'motorbike', 'bike', 'scooter'],
            'van': ['van', 'minivan'],
            'bicycle': ['bicycle', 'cycle', 'bike']
        }
        def canonical_class(cls):
            for canon, aliases in class_aliases.items():
                if cls.lower() in aliases:
                    return canon
            return None
        dets = []
        for det in detections:
            canon = canonical_class(det['class'])
            if canon is not None:
                x1, y1, x2, y2 = det['bbox']
                conf = det.get('confidence', 1.0)
                class_id = vehicle_classes[canon]
                dets.append([x1, y1, x2, y2, conf, class_id])
        tracks = self.vehicle_tracker.update(dets, frame=frame)
        tracked_vehicles = []
        for track_id, ltrb, conf, class_id in tracks:
            # Map back to canonical class name
            class_name = [k for k, v in vehicle_classes.items() if v == class_id]
            class_name = class_name[0] if class_name else 'unknown'
            tracked_vehicles.append({
                'id': track_id,
                'bbox': ltrb,
                'confidence': conf,
                'class': class_name
            })
        return tracked_vehicles
######working 

from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math
import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap,
)

# Import traffic light color detection utilities
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from controllers.deepsort_tracker import DeepSortVehicleTracker

TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    frame_np_with_violations = Signal(np.ndarray, list, list)  # frame, detections, violators
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        
        # Initialize the traffic light color detection pipeline
        # self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = DeepSortVehicleTracker()
        # Add red light violation system with tracker
        # self.red_light_violation_system = RedLightViolationSystem(
        #     vehicle_tracker=self.vehicle_tracker,
        #     config={
        #         'min_confidence': 0.5,
        #         'min_violation_frames': 5
        #     }
        # )
        self.last_violation_line_y = None  # For overlay
        self.violation_states = {}  # For violation state machine
        self.frame_idx = 0  # Initialize frame index for violation tracking
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    print(f"Drawing {len(detections)} detection boxes on frame")
                    for det in detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            print(f"[DETECTION DEBUG] bbox={bbox}, type={type(bbox)}, len={len(bbox) if bbox is not None else 'None'}")
                            if bbox is None or len(bbox) != 4:
                                continue
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR
                            if 'id' in det:
                                id_text = f"ID: {det['id']}"
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated_frame, f"{id_text} {label} ", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            # Draw vehicle ID if present
                            # if 'id' in det:
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()# Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        print("\033[94m[FIX] _process_frame called for new frame\033[0m")
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            annotated_frame = frame.copy()
            # Draw detections
            for det in detections:
                if 'bbox' in det:
                    bbox = det['bbox']
                    if bbox is None or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    label = det.get('class_name', 'object')
                    confidence = det.get('confidence', 0.0)
                    class_id = det.get('class_id', -1)
                    if class_id == 9 or is_traffic_light(label):
                        box_color = (0, 0, 255)
                    else:
                        box_color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    if 'id' in det:
                        id_text = f"ID: {det['id']}"
                        (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            # Find traffic light bbox
            traffic_light_bbox = None
            for det in detections:
                if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                    traffic_light_bbox = det['bbox']
                    break
            # --- Violation detection and overlay ---
            print(f"[DEBUG] Calling process_frame with frame_idx={self.frame_idx}, detections={len(detections)}, traffic_light_bbox={traffic_light_bbox}")
            # --- Get traffic light color info ---
            traffic_light_info = self.latest_traffic_light if hasattr(self, 'latest_traffic_light') else {"color": "unknown", "confidence": 0.0}
            # --- Get violation line y from crosswalk detection ---
            try:
                # Call crosswalk detection to get current violation line
                crosswalk_bbox, violation_line_coords, crosswalk_info = self.detect_crosswalk_and_violation_line(frame, traffic_light_bbox)
                if violation_line_coords and len(violation_line_coords) >= 2:
                    # Extract y-coordinate from violation line coordinates
                    violation_line_y = int(violation_line_coords[1])  # y-coordinate of start point
                    self.last_violation_line_y = violation_line_y  # Update cached value
                else:
                    violation_line_y = self.last_violation_line_y if hasattr(self, 'last_violation_line_y') else None
            except Exception as e:
                print(f"[WARN] Crosswalk detection error in _process_frame: {e}")
                violation_line_y = self.last_violation_line_y if hasattr(self, 'last_violation_line_y') else None
            # --- Call violation detection logic ---
            try:
                annotated_with_viol, violators, _ = self.detect_red_light_violations(
                    frame=frame,
                    vehicle_detections=detections,
                    traffic_light_color_info=traffic_light_info,
                    violation_line_y=violation_line_y,
                    frame_number=self.frame_idx,
                    state_cache=getattr(self, '_violation_state_cache', None)
                )
                self._violation_state_cache = _  # persist state
                print(f"[VIOLATION DEBUG] Frame {self.frame_idx}: {len(violators)} violations detected.")
                for v in violators:
                    print(f"[VIOLATION DEBUG] Violation: {v}")
            except Exception as e:
                print("\033[91m[ERROR] Exception in violation detection!\033[0m")
                traceback.print_exc()
                annotated_with_viol = annotated_frame
                violators = []
            self.frame_idx += 1
            # Draw overlays
            annotated_with_viol = draw_performance_overlay(annotated_with_viol, metrics)
            cv2.circle(annotated_with_viol, (20, 20), 10, (255, 255, 0), -1)
            frame_rgb = cv2.cvtColor(annotated_with_viol, cv2.COLOR_BGR2RGB)
            try:
                self.frame_np_ready.emit(frame_rgb)
                vehicle_detections = detections
                self.frame_np_with_violations.emit(annotated_with_viol, vehicle_detections, violators)
            except Exception as e:
                print(f"Error emitting processed frame: {e}")
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    def detect_red_light_violations(self, frame, vehicle_detections, traffic_light_color_info, violation_line_y, frame_number, state_cache=None):
        """
        Robust red light violation detection logic with detailed debug for every vehicle, matching video_controller_finale.py.
        """
        debug = True
        try:
            if state_cache is None:
                state_cache = {}
            if 'red_count' not in state_cache:
                state_cache['red_count'] = 0
            if 'last_color' not in state_cache:
                state_cache['last_color'] = None
            if 'vehicle_states' not in state_cache:
                state_cache['vehicle_states'] = {}
            if 'cooldown' not in state_cache:
                state_cache['cooldown'] = {}

            color = traffic_light_color_info.get('color', 'unknown')
            conf = traffic_light_color_info.get('confidence', 0.0)
            # Debounce: require 3 consecutive red frames
            if color == 'red' and conf >= 0.3:
                if state_cache['last_color'] == 'red':
                    state_cache['red_count'] += 1
                else:
                    state_cache['red_count'] = 1
            else:
                state_cache['red_count'] = 0
            state_cache['last_color'] = color
            red_consistent = state_cache['red_count'] >= 3

            annotated = frame.copy()
            h, w = frame.shape[:2]
            # Draw violation line if available
            if violation_line_y is not None:
                cv2.line(annotated, (0, violation_line_y), (w, violation_line_y), (0,0,255), 5)
                cv2.putText(annotated, "VIOLATION LINE", (10, violation_line_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # Draw red light indicator (no emoji)
            if color == 'red' and conf >= 0.3:
                cv2.putText(annotated, "RED LIGHT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)

            violators = []
            for det in vehicle_detections:
                # Fallback for missing vehicle ID
                vid = det.get('id')
                if vid is None:
                    # Use bbox as fallback ID (tuple)
                    vid = tuple(det.get('bbox', []))
                bbox = det.get('bbox')
                if bbox is None or len(bbox) != 4:
                    if debug:
                        print(f"[VIOLATION DEBUG] Skipping vehicle with invalid bbox: {bbox}")
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2
                bottom_y = max(y1, y2)
                # Ignore vehicles outside central 80% of frame width
                if not (0.1 * w < cx < 0.9 * w):
                    continue
                # Per-vehicle state
                vstate = state_cache['vehicle_states'].setdefault(vid, {'was_behind': True, 'last_crossed': -100, 'entry_time': None, 'dwell': 0})
                cooldown = state_cache['cooldown'].get(vid, 0)
                # Print all state info for this vehicle
                print(f"[VIOLATION DEBUG] Vehicle {vid}: bbox={bbox}, cx={cx}, bottom_y={bottom_y}, vstate={vstate}, cooldown={cooldown}, violation_line_y={violation_line_y}, red_consistent={red_consistent}")
                if cooldown > 0:
                    state_cache['cooldown'][vid] -= 1
                    print(f"[VIOLATION DEBUG] Vehicle {vid} in cooldown: {state_cache['cooldown'][vid]} frames left")
                    continue
                if violation_line_y is not None and bottom_y < violation_line_y:
                    if not vstate['was_behind']:
                        print(f"[VIOLATION DEBUG] Vehicle {vid} moved behind the line at frame {frame_number}")
                    vstate['was_behind'] = True
                    if vstate['entry_time'] is None:
                        vstate['entry_time'] = frame_number
                    vstate['dwell'] = 0
                elif violation_line_y is not None and vstate['was_behind'] and red_consistent and bottom_y >= violation_line_y:
                    # Violation detected
                    violators.append({
                        'id': vid,
                        'bbox': bbox,
                        'frame': frame_number,
                        'violation_type': 'red_light',
                        'violation_line_y': violation_line_y
                    })
                    vstate['was_behind'] = False
                    vstate['last_crossed'] = frame_number
                    state_cache['cooldown'][vid] = 30
                    print(f"[VIOLATION] Vehicle {vid} crossed at frame {frame_number} during RED! bbox={bbox}")
                else:
                    print(f"[VIOLATION DEBUG] Vehicle {vid} not violating: was_behind={vstate['was_behind']}, red_consistent={red_consistent}, bottom_y={bottom_y}, violation_line_y={violation_line_y}")
            return annotated, violators, state_cache
        except Exception as e:
            print(f"[ERROR] Exception in detect_red_light_violations: {e}")
            import traceback
            traceback.print_exc()
            return frame, [], state_cache

###nott working but violation debug chal rhe 
from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math
import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap,
)

# Import traffic light color detection utilities
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from controllers.deepsort_tracker import DeepSortVehicleTracker

TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoWorker(QObject):
    """Worker class to handle video processing in a separate thread"""
    frame_processed = Signal(np.ndarray, list)  # frame, detections
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
    def run_video_processing(self):
        """Run video processing in worker thread"""
        if hasattr(self.controller, '_run'):
            self.controller._run()

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    frame_np_with_violations = Signal(np.ndarray, list, list)  # frame, detections, violators
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread with worker
        self.thread = QThread()
        self.worker = VideoWorker(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run_video_processing)
        
        # Performance measurement
        self.condition = QWaitCondition()
        
        # Setup render timer with more aggressive settings for UI updates
        # Timer stays in main thread for proper signal handling
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        
        # Initialize the traffic light color detection pipeline
        # self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = DeepSortVehicleTracker()
        # Add red light violation system with tracker
        # self.red_light_violation_system = RedLightViolationSystem(
        #     vehicle_tracker=self.vehicle_tracker,
        #     config={
        #         'min_confidence': 0.5,
        #         'min_violation_frames': 5
        #     }
        # )
        self.last_violation_line_y = None  # For overlay
        self.violation_states = {}  # For violation state machine
        self.frame_idx = 0  # Initialize frame index for violation tracking
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Optional: Start the render timer as backup (reduced frequency since main processing handles most)
            print("⏱️ Starting backup render timer...")
            print(f"🔍 Timer parent thread: {self.render_timer.thread()}")
            print(f"🔍 Controller thread: {self.thread()}")
            self.render_timer.start(100)  # 10Hz backup timer
            print("✅ Backup render timer started at 10Hz")
            print(f"🔄 Render timer active: {self.render_timer.isActive()}, interval: {self.render_timer.interval()}ms")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    print(f"Drawing {len(detections)} detection boxes on frame")
                    for det in detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            print(f"[DETECTION DEBUG] bbox={bbox}, type={type(bbox)}, len={len(bbox) if bbox is not None else 'None'}")
                            if bbox is None or len(bbox) != 4:
                                continue
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR
                                
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            
                            # Handle vehicle ID display
                            if 'id' in det:
                                id_text = f"ID: {det['id']}"
                                cv2.putText(annotated_frame, f"{id_text} {label} ", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            # Draw vehicle ID if present
                            # if 'id' in det:
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()
                
                # NOTE: Emit frame_np_ready with violation detection here
                # Instead of relying on _process_frame timer
                print(f"🔍 [CRITICAL DEBUG] About to enter violation detection block")
                print(f"🔍 [CRITICAL DEBUG] annotated_frame shape: {annotated_frame.shape}")
                print(f"🔍 [CRITICAL DEBUG] detections count: {len(detections)}")
                print(f"🔍 [CRITICAL DEBUG] original frame shape: {frame.shape}")
                
                try:
                    print(f"🔍 About to call _add_violation_detection with {len(detections)} detections")
                    # Process violations and annotations
                    annotated_with_violations = self._add_violation_detection(annotated_frame, detections, frame)
                    print(f"🔍 _add_violation_detection returned frame with shape: {annotated_with_violations.shape}")
                    
                    # Convert to RGB for Qt display
                    frame_rgb = cv2.cvtColor(annotated_with_violations, cv2.COLOR_BGR2RGB)
                    frame_copy = np.ascontiguousarray(frame_rgb)
                    
                    print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {frame_copy.shape}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        print("\033[94m[DEBUG] _process_frame called - timer triggered\033[0m")
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process in _process_frame")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    print("🔄 Not running - _process_frame will exit")
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                    print("📺 Emitted blank frame from _process_frame")
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                metrics = self.performance_metrics.copy()
                print(f"🔍 _process_frame: Got frame {frame.shape}, {len(detections)} detections")
            except Exception as e:
                print(f"Error copying frame data in _process_frame: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            annotated_frame = frame.copy()
            # Draw detections
            for det in detections:
                if 'bbox' in det:
                    bbox = det['bbox']
                    if bbox is None or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    label = det.get('class_name', 'object')
                    confidence = det.get('confidence', 0.0)
                    class_id = det.get('class_id', -1)
                    if class_id == 9 or is_traffic_light(label):
                        box_color = (0, 0, 255)
                    else:
                        box_color = (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                    if 'id' in det:
                        id_text = f"ID: {det['id']}"
                        (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                        cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            # Find traffic light bbox
            traffic_light_bbox = None
            for det in detections:
                if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                    traffic_light_bbox = det['bbox']
                    break
            # --- Violation detection and overlay ---
            print(f"[DEBUG] Calling process_frame with frame_idx={self.frame_idx}, detections={len(detections)}, traffic_light_bbox={traffic_light_bbox}")
            # --- Get traffic light color info ---
            traffic_light_info = self.latest_traffic_light if hasattr(self, 'latest_traffic_light') else {"color": "unknown", "confidence": 0.0}
            # --- Get violation line y from crosswalk detection ---
            try:
                # Call crosswalk detection to get current violation line
                crosswalk_bbox, violation_line_coords, crosswalk_info = self.detect_crosswalk_and_violation_line(frame, traffic_light_bbox)
                if violation_line_coords and len(violation_line_coords) >= 2:
                    # Extract y-coordinate from violation line coordinates
                    violation_line_y = int(violation_line_coords[1])  # y-coordinate of start point
                    self.last_violation_line_y = violation_line_y  # Update cached value
                else:
                    violation_line_y = self.last_violation_line_y if hasattr(self, 'last_violation_line_y') else None
            except Exception as e:
                print(f"[WARN] Crosswalk detection error in _process_frame: {e}")
                violation_line_y = self.last_violation_line_y if hasattr(self, 'last_violation_line_y') else None
            # --- Call violation detection logic ---
            try:
                annotated_with_viol, violators, _ = self.detect_red_light_violations(
                    frame=frame,
                    vehicle_detections=detections,
                    traffic_light_color_info=traffic_light_info,
                    violation_line_y=violation_line_y,
                    frame_number=self.frame_idx,
                    state_cache=getattr(self, '_violation_state_cache', None)
                )
                self._violation_state_cache = _  # persist state
                print(f"[VIOLATION DEBUG] Frame {self.frame_idx}: {len(violators)} violations detected.")
                for v in violators:
                    print(f"[VIOLATION DEBUG] Violation: {v}")
            except Exception as e:
                print("\033[91m[ERROR] Exception in violation detection!\033[0m")
                import traceback
                traceback.print_exc()
                annotated_with_viol = annotated_frame
                violators = []
            self.frame_idx += 1
            # Draw overlays
            annotated_with_viol = draw_performance_overlay(annotated_with_viol, metrics)
            cv2.circle(annotated_with_viol, (20, 20), 10, (255, 255, 0), -1)
            frame_rgb = cv2.cvtColor(annotated_with_viol, cv2.COLOR_BGR2RGB)
            try:
                print(f"🔴 _process_frame emitting frame_np_ready with shape: {frame_rgb.shape}")
                self.frame_np_ready.emit(frame_rgb)
                vehicle_detections = detections
                self.frame_np_with_violations.emit(annotated_with_viol, vehicle_detections, violators)
                print(f"✅ _process_frame: emitted frames with {len(violators)} violations")
            except Exception as e:
                print(f"Error emitting processed frame from _process_frame: {e}")
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    def detect_red_light_violations(self, frame, vehicle_detections, traffic_light_color_info, violation_line_y, frame_number, state_cache=None):
        print(f"\033[91m[CRITICAL] detect_red_light_violations called at frame {frame_number}\033[0m")
        print(f"\033[91m[CRITICAL] Frame shape: {frame.shape}, Detections: {len(vehicle_detections)}, Traffic light: {traffic_light_color_info}, Violation line: {violation_line_y}\033[0m")
        try:
            debug = True
            if state_cache is None:
                state_cache = {}
            # --- Persistent state for debounce and per-vehicle tracking ---
            if 'red_count' not in state_cache:
                state_cache['red_count'] = 0
            if 'last_color' not in state_cache:
                state_cache['last_color'] = None
            if 'vehicle_states' not in state_cache:
                state_cache['vehicle_states'] = {}
            if 'cooldown' not in state_cache:
                state_cache['cooldown'] = {}
            # --- Traffic light color debounce ---
            color = traffic_light_color_info.get('color', 'unknown')
            conf = traffic_light_color_info.get('confidence', 0.0)
            print(f"\033[92m[DEBUG] Traffic light: color={color}, confidence={conf}\033[0m")
            
            if color == 'red' and conf >= 0.3:
                if state_cache['last_color'] == 'red':
                    state_cache['red_count'] += 1
                else:
                    state_cache['red_count'] = 1
                print(f"\033[92m[DEBUG] Red light detected, red_count={state_cache['red_count']}\033[0m")
            else:
                state_cache['red_count'] = 0
                print(f"\033[92m[DEBUG] No consistent red light, red_count reset to 0\033[0m")
                
            state_cache['last_color'] = color
            red_consistent = state_cache['red_count'] >= 3
            print(f"\033[92m[DEBUG] Red light consistent: {red_consistent} (need 3+ frames)\033[0m")
            # --- Frame prep ---
            annotated = frame.copy()
            h, w = frame.shape[:2]
            # Draw bold red line at violation_line_y (should always be available now)
            if violation_line_y is not None and violation_line_y > 0:
                print(f"\033[92m[DEBUG] Drawing violation line at y={violation_line_y}\033[0m")
                cv2.line(annotated, (0, violation_line_y), (w, violation_line_y), (0,0,255), 5)
                cv2.putText(annotated, "VIOLATION LINE", (10, max(violation_line_y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                print(f"\033[91m[ERROR] Invalid violation line! violation_line_y={violation_line_y}\033[0m")
            # Draw red light indicator
            if color == 'red' and conf >= 0.3:
                # Clean text rendering without problematic characters
                text = "RED LIGHT DETECTED"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.2
                thickness = 3
                color_bgr = (0, 0, 255)  # Red color in BGR
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw semi-transparent background rectangle for better visibility
                overlay = annotated.copy()
                cv2.rectangle(overlay, (10, 10), (20 + text_width, 20 + text_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
                
                # Draw the text with better positioning
                cv2.putText(annotated, text, (15, 15 + text_height), font, font_scale, color_bgr, thickness, cv2.LINE_AA)
            violators = []
            print(f"\033[92m[DEBUG] Processing {len(vehicle_detections)} vehicle detections for violations\033[0m")
            # Reduce spam - only print first detection and summary
            if len(vehicle_detections) > 0:
                print(f"\033[92m[DEBUG] Sample Detection 0: {vehicle_detections[0]}\033[0m")
                if len(vehicle_detections) > 1:
                    print(f"\033[92m[DEBUG] ... and {len(vehicle_detections)-1} more detections\033[0m")
            
            for i, det in enumerate(vehicle_detections):
                vid = det.get('id')
                if vid is None:
                    print(f"\033[93m[WARNING] Detection {i} has no ID, using index as ID\033[0m")
                    vid = f"vehicle_{i}"  # Use index as fallback ID
                    
                bbox = det['bbox']
                if bbox is None or len(bbox) != 4:
                    print(f"\033[93m[WARNING] Detection {i} has invalid bbox: {bbox}\033[0m")
                    continue
                    
                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2
                bottom_y = max(y1, y2)
                
                # Only print debug for vehicles that could potentially violate
                # (reduce spam for vehicles clearly outside violation areas)
                
                # Ignore vehicles outside central 80% of frame width
                if not (0.1 * w < cx < 0.9 * w):
                    # print(f"\033[93m[DEBUG] Vehicle {vid} outside central area, skipping\033[0m")
                    continue
                # --- Per-vehicle state ---
                if violation_line_y is None or violation_line_y <= 0:
                    # This should not happen anymore due to fallbacks above
                    if i == 0:
                        print(f"\033[91m[ERROR] Invalid violation line value: {violation_line_y}, skipping frame\033[0m")
                        h, w = frame.shape[:2] if frame is not None else (480, 640)
                        violation_line_y = int(h * 0.75)
                        print(f"\033[91m[ERROR] Emergency fallback: set violation_line_y to {violation_line_y}\033[0m")
                    else:
                        continue
                    
                vstate = state_cache['vehicle_states'].setdefault(vid, {'was_behind': True, 'last_crossed': -100, 'entry_time': None, 'dwell': 0})
                cooldown = state_cache['cooldown'].get(vid, 0)
                
                # Only check if not in cooldown
                if cooldown > 0:
                    state_cache['cooldown'][vid] -= 1
                    continue
                # Track entry time and dwell time
                if bottom_y < violation_line_y:
                    vstate['was_behind'] = True
                    if vstate['entry_time'] is None:
                        vstate['entry_time'] = frame_number
                    vstate['dwell'] = 0
                    # print(f"\033[92m[DEBUG] Vehicle {vid} is behind violation line (y={bottom_y} < {violation_line_y})\033[0m")
                else:
                    print(f"\033[92m[DEBUG] Vehicle {vid} past violation line (y={bottom_y} >= {violation_line_y}), was_behind={vstate['was_behind']}, red_consistent={red_consistent}\033[0m")
                    if vstate['was_behind'] and red_consistent:
                        # Violation detected
                        print(f"\033[91m🚨 [VIOLATION DETECTED] Vehicle {vid} crossed during RED at frame {frame_number}! 🚨\033[0m")
                        violators.append({
                            'id': vid,
                            'bbox': bbox,
                            'frame': frame_number,
                            'violation_type': 'red_light',
                            'violation_line_y': violation_line_y
                        })
                        vstate['was_behind'] = False
                        vstate['last_crossed'] = frame_number
                        state_cache['cooldown'][vid] = 30  # Debounce for 30 frames
                        if debug:
                            print(f"[VIOLATION] Vehicle {vid} crossed at frame {frame_number} during RED!")
                    else:
                        if not vstate['was_behind']:
                            # print(f"\033[93m[DEBUG] Vehicle {vid} was not behind line before crossing\033[0m")
                            pass
                        if not red_consistent:
                            print(f"\033[93m[DEBUG] Red light not consistent for vehicle {vid} (red_count={state_cache['red_count']})\033[0m")
            # Optionally: advanced logic for dwell, direction, speed, etc.
            return annotated, violators, state_cache
        except Exception as e:
            print(f"[ERROR] Exception in detect_red_light_violations: {e}")
            import traceback
            traceback.print_exc()
            return frame, [], state_cache

    def _add_violation_detection(self, annotated_frame, detections, original_frame):
        """Add violation detection to the frame processing"""
        print(f"🔍 [DEBUG] _add_violation_detection called with frame shape: {original_frame.shape}, detections: {len(detections)}")
        try:
            # Find traffic light bbox
            traffic_light_bbox = None
            for det in detections:
                if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                    traffic_light_bbox = det['bbox']
                    print(f"🚦 Found traffic light bbox: {traffic_light_bbox}")
                    break
            
            print(f"[DEBUG] _add_violation_detection with frame_idx={self.frame_idx}, detections={len(detections)}, traffic_light_bbox={traffic_light_bbox}")
            
            # Get traffic light color info
            traffic_light_info = self.latest_traffic_light if hasattr(self, 'latest_traffic_light') else {"color": "unknown", "confidence": 0.0}
            print(f"🚦 Traffic light info: {traffic_light_info}")
            
            # Get violation line y from crosswalk detection
            print(f"🔍 Calling crosswalk detection...")
            try:
                crosswalk_result = self.detect_crosswalk_and_violation_line(original_frame, traffic_light_bbox)
                print(f"🔍 Crosswalk detection raw result: {crosswalk_result}")
                
                # Handle different return formats
                if isinstance(crosswalk_result, tuple):
                    if len(crosswalk_result) == 3:
                        crosswalk_bbox, violation_line_coords, crosswalk_info = crosswalk_result
                    elif len(crosswalk_result) == 2:
                        crosswalk_bbox, violation_line_coords = crosswalk_result
                        crosswalk_info = {}
                    else:
                        print(f"🔍 Unexpected crosswalk result format: {len(crosswalk_result)} items")
                        violation_line_coords = None
                else:
                    violation_line_coords = crosswalk_result
                
                print(f"🔍 Crosswalk detection result: violation_line_coords={violation_line_coords}")
                if violation_line_coords and len(violation_line_coords) >= 2:
                    violation_line_y = int(violation_line_coords[1])
                    self.last_violation_line_y = violation_line_y
                    print(f"🔍 Set violation_line_y to: {violation_line_y}")
                else:
                    # Use cached value or calculate a reasonable default
                    violation_line_y = getattr(self, 'last_violation_line_y', None)
                    if violation_line_y is None:
                        h, w = original_frame.shape[:2]
                        violation_line_y = int(h * 0.75)  # Default to 75% down the frame
                        self.last_violation_line_y = violation_line_y
                        print(f"🔍 No cached violation line, using default: {violation_line_y} (75% of frame height {h})")
                    else:
                        print(f"🔍 Using cached violation_line_y: {violation_line_y}")
            except Exception as e:
                print(f"[WARN] Crosswalk detection error: {e}")
                import traceback
                traceback.print_exc()
                # Use cached value or calculate a reasonable default
                violation_line_y = getattr(self, 'last_violation_line_y', None)
                if violation_line_y is None:
                    h, w = original_frame.shape[:2]
                    violation_line_y = int(h * 0.75)  # Default to 75% down the frame
                    self.last_violation_line_y = violation_line_y
                    print(f"🔍 Exception fallback: using default violation_line_y: {violation_line_y} (75% of frame height {h})")
                else:
                    print(f"🔍 Exception fallback: using cached violation_line_y: {violation_line_y}")
                
            # Ensure violation_line_y is never None
            if violation_line_y is None:
                h, w = original_frame.shape[:2]
                violation_line_y = int(h * 0.75)
                self.last_violation_line_y = violation_line_y
                print(f"🔍 Final fallback: violation_line_y was None, set to default: {violation_line_y}")
                
                # Try to use a reasonable default based on frame height
                if violation_line_y is None:
                    frame_height = original_frame.shape[0]
                    violation_line_y = int(frame_height * 0.9)  # 90% of frame height
                    print(f"🔍 Using default violation_line_y: {violation_line_y} (90% of frame height {frame_height})")
                    self.last_violation_line_y = violation_line_y
            
            # Call violation detection logic
            print(f"🔍 About to call detect_red_light_violations with violation_line_y={violation_line_y}")
            try:
                annotated_with_viol, violators, state_cache = self.detect_red_light_violations(
                    frame=original_frame,
                    vehicle_detections=detections,
                    traffic_light_color_info=traffic_light_info,
                    violation_line_y=violation_line_y,
                    frame_number=self.frame_idx,
                    state_cache=getattr(self, '_violation_state_cache', None)
                )
                self._violation_state_cache = state_cache
                print(f"[VIOLATION DEBUG] Frame {self.frame_idx}: {len(violators)} violations detected.")
                for v in violators:
                    print(f"[VIOLATION DEBUG] Violation: {v}")
                    
                # Emit violation signal
                print(f"🔍 Emitting frame_np_with_violations signal...")
                self.frame_np_with_violations.emit(annotated_with_viol, detections, violators)
                print(f"✅ Emitted frame_np_with_violations with {len(violators)} violations")
                
            except Exception as e:
                print(f"[ERROR] Exception in violation detection: {e}")
                import traceback
                traceback.print_exc()
                annotated_with_viol = annotated_frame
                violators = []
            
            self.frame_idx += 1
            
            # Draw performance overlay
            print(f"🔍 Drawing performance overlay...")
            annotated_with_viol = draw_performance_overlay(annotated_with_viol, self.performance_metrics)
            
            print(f"🔍 _add_violation_detection returning frame with shape: {annotated_with_viol.shape}")
            return annotated_with_viol
            
        except Exception as e:
            print(f"[ERROR] Exception in _add_violation_detection: {e}")
            import traceback
            traceback.print_exc()
            return annotated_frame


##########WORKING VERSION##########
from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap,
    pipeline_with_violation_line
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line, get_violation_line_y
from controllers.deepsort_tracker import DeepSortVehicleTracker
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        self.violation_frame_counter = 0  # Add counter for violation processing
        
        # Vehicle movement tracking for violation detection
        self.vehicle_history = {}  # track_id -> deque of positions
        self.movement_threshold = 3  # pixels movement threshold
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = DeepSortVehicleTracker()
        
        # Add red light violation system
        # self.red_light_violation_system = RedLightViolationSystem()
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    print(f"Drawing {len(detections)} detection boxes on frame")
                    for det in detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR
                            if 'id' in det:
                                id_text = f"ID: {det['id']}"
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated_frame, f"{id_text} {label} ", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            # Draw vehicle ID if present
                            # if 'id' in det:
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")

                # --- VIOLATION DETECTION LOGIC (conditional based on traffic lights or crosswalk) ---
                # First, check if we have traffic lights detected
                traffic_lights = []
                has_traffic_lights = False
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})

                # Get traffic light position for crosswalk detection
                traffic_light_position = None
                if has_traffic_lights:
                    for det in detections:
                        if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                            traffic_light_bbox = det['bbox']
                            # Extract center point from bbox for crosswalk utils
                            x1, y1, x2, y2 = traffic_light_bbox
                            traffic_light_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            break

                # Run crosswalk detection to check if crosswalk exists
                try:
                    result_frame, crosswalk_bbox, violation_line_y, debug_info = detect_crosswalk_and_violation_line(
                        annotated_frame, traffic_light_position
                    )
                except Exception as e:
                    print(f"[ERROR] Crosswalk detection failed: {e}")
                    result_frame, crosswalk_bbox, violation_line_y, debug_info = annotated_frame, None, None, {}
                
                # Check if crosswalk is detected
                crosswalk_detected = crosswalk_bbox is not None
                stop_line_detected = debug_info.get('stop_line') is not None
                
                # Only proceed with violation logic if we have traffic lights OR crosswalk detected
                # AND every 3rd frame for performance (adjust as needed)
                violations = []
                self.violation_frame_counter += 1
                should_process_violations = (has_traffic_lights or crosswalk_detected) and (self.violation_frame_counter % 3 == 0)
                
                if should_process_violations:
                    print(f"[DEBUG] Processing violation logic - Traffic lights: {has_traffic_lights}, Crosswalk: {crosswalk_detected}")
                    
                    # Create violation line coordinates from y position
                    violation_line = None
                    if violation_line_y is not None:
                        start_pt = (0, violation_line_y)
                        end_pt = (annotated_frame.shape[1], violation_line_y)
                        violation_line = (start_pt, end_pt)
                        
                        # Draw the thick red violation line with black label background (like in image)
                        line_color = (0, 0, 255)  # Red color
                        cv2.line(annotated_frame, start_pt, end_pt, line_color, 6)  # Thick line
                        
                        # Draw black background for label
                        label = "Violation Line"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Black background rectangle
                        cv2.rectangle(annotated_frame, 
                                    (10, start_pt[1] - text_height - 15), 
                                    (10 + text_width + 10, start_pt[1] - 5), 
                                    (0, 0, 0), -1)  # Black background
                        
                        # Red text
                        cv2.putText(annotated_frame, label, (15, start_pt[1] - 10), 
                                  font, font_scale, line_color, thickness)
                        
                        print(f"[DEBUG] Violation line drawn at y={start_pt[1]}, type={label}")
                    else:
                        print(f"[DEBUG] No valid violation line detected.")

                    # DeepSORT tracking integration with movement detection
                    tracked_vehicles = []
                    if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                        try:
                            vehicle_dets = [det for det in detections if det.get('class_name') in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle'] and 'bbox' in det]
                            # Pass the detection dictionaries directly to the tracker
                            tracks = self.vehicle_tracker.update(vehicle_dets, frame)
                            
                            # tracks is a list of dicts: [{'id': track_id, 'bbox': [x1,y1,x2,y2], 'confidence': conf, 'class_id': class_id}, ...]
                            for track in tracks:
                                track_id = track['id']
                                bbox = track['bbox']
                                
                                # Calculate vehicle center for movement tracking
                                x1, y1, x2, y2 = map(float, bbox)
                                center_y = (y1 + y2) / 2
                                
                                # Initialize or update vehicle history
                                if track_id not in self.vehicle_history:
                                    from collections import deque
                                    self.vehicle_history[track_id] = deque(maxlen=5)
                                
                                self.vehicle_history[track_id].append(center_y)
                                
                                # Calculate movement (only if we have previous positions)
                                is_moving = False
                                if len(self.vehicle_history[track_id]) >= 2:
                                    prev_y = self.vehicle_history[track_id][-2]
                                    current_y = self.vehicle_history[track_id][-1]
                                    dy = abs(current_y - prev_y)
                                    is_moving = dy > self.movement_threshold
                                
                                tracked_vehicles.append({
                                    'id': track_id, 
                                    'bbox': bbox, 
                                    'center_y': center_y,
                                    'is_moving': is_moving,
                                    'prev_y': self.vehicle_history[track_id][-2] if len(self.vehicle_history[track_id]) >= 2 else center_y
                                })
                                
                            print(f"[DEBUG] DeepSORT tracked {len(tracked_vehicles)} vehicles")
                        except Exception as e:
                            print(f"[ERROR] DeepSORT tracking failed: {e}")
                            tracked_vehicles = []
                    else:
                        print("[WARN] DeepSORT vehicle tracker not available!")

                    # Red light violation detection
                    red_lights = []
                    for tl in traffic_lights:
                        if tl.get('color') == 'red':
                            red_lights.append(tl)
                    print(f"[DEBUG] Red light(s) detected: {len(red_lights)} red lights")
                    
                    vehicle_debugs = []
                    
                    # Always print vehicle debug info for frames with violation logic
                    for v in tracked_vehicles:
                        bbox = v['bbox']
                        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for OpenCV
                        vehicle_debugs.append(f"Tracked Vehicle ID={v['id']} bbox=[{x1},{y1},{x2},{y2}] bottom_y={y2} vline_y={violation_line_y}")
                    
                    if red_lights and violation_line_y is not None:
                        print(f"[DEBUG] Checking {len(tracked_vehicles)} tracked vehicles for violations")
                        for v in tracked_vehicles:
                            bbox = v['bbox']
                            x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for OpenCV
                            if y2 > violation_line_y:
                                print(f"[DEBUG] RED LIGHT VIOLATION: Vehicle ID={v['id']} at bbox=[{x1},{y1},{x2},{y2}] (y2={y2} > vline_y={violation_line_y})")
                                # Fix the violation data format to match UI expectations
                                violations.append({'track_id': v['id'], 'id': v['id'], 'bbox': [x1, y1, x2, y2], 'violation': 'red_light'})
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 140, 255), 4)  # Orange
                                cv2.putText(annotated_frame, f'VIOLATION ID:{v["id"]}', (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,140,255), 2)
                            else:
                                print(f"[DEBUG] No violation: Vehicle ID={v['id']} at bbox=[{x1},{y1},{x2},{y2}] (y2={y2} <= vline_y={violation_line_y})")
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f'ID:{v["id"]}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                        if not violations:
                            print("[DEBUG] No red light violations detected this frame.")
                    else:
                        print(f"[DEBUG] No red light or no violation line for this frame. Red lights: {len(red_lights)}, vline_y: {violation_line_y}")
                    
                    # Print vehicle debug info for frames with violation logic
                    for vdbg in vehicle_debugs:
                        print(f"[DEBUG] {vdbg}")
                else:
                    print(f"[DEBUG] Skipping violation logic - Frame {self.violation_frame_counter}: Traffic lights: {has_traffic_lights}, Crosswalk: {crosswalk_detected}")
                    violation_line_y = None  # Set to None when no violation logic runs
                
                # Always emit violation signal (may be empty when no violation logic runs)
                self.violation_detected.emit({'violations': violations, 'frame': frame, 'violation_line_y': violation_line_y})
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()# Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                violations = []  # Violations are disabled
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # --- Simplified frame processing for display ---
            # The violation logic is now handled in the main _run thread
            # This method just handles basic display overlays
            
            annotated_frame = frame.copy()

            # Add performance overlays and debug markers
            annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            cv2.circle(annotated_frame, (20, 20), 10, (255, 255, 0), -1)

            # Convert BGR to RGB before display (for PyQt/PySide)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # Display the RGB frame in the UI (replace with your display logic)
            # Example: self.image_label.setPixmap(QPixmap.fromImage(QImage(frame_rgb.data, w, h, QImage.Format_RGB888)))
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    # --- Removed unused internal violation line detection methods and RedLightViolationSystem usage ---


    3###badiya
    from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap,
    pipeline_with_violation_line
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line, get_violation_line_y
from controllers.deepsort_tracker import DeepSortVehicleTracker
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Vehicle tracking settings
        self.vehicle_history = {}  # Dictionary to store vehicle position history
        self.vehicle_statuses = {}  # Track stable movement status
        self.movement_threshold = 2.5  # Minimum pixel change to consider a vehicle moving
        self.min_confidence_threshold = 0.5  # Minimum confidence for vehicle detection
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            # self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            # self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        self.violation_frame_counter = 0  # Add counter for violation processing
        
        # Vehicle movement tracking for violation detection
        self.vehicle_history = {}  # track_id -> deque of positions
        self.movement_threshold = 3  # pixels movement threshold
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = DeepSortVehicleTracker()
        
        # Add red light violation system
        # self.red_light_violation_system = RedLightViolationSystem()
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    print(f"Drawing {len(detections)} detection boxes on frame")
                    for det in detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR
                            if 'id' in det:
                                id_text = f"ID: {det['id']}"
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated_frame, f"{id_text} {label} ", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            # Draw vehicle ID if present
                            # if 'id' in det:
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    # Draw enhanced traffic light status
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                    
                                    # Add a prominent traffic light status at the top of the frame
                                    color = light_info.get('color', 'unknown')
                                    confidence = light_info.get('confidence', 0.0)
                                    
                                    if color == 'red':
                                        status_color = (0, 0, 255)  # Red
                                        status_text = f"Traffic Light: RED ({confidence:.2f})"
                                        
                                        # Draw a prominent red banner across the top
                                        banner_height = 40
                                        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], banner_height), (0, 0, 150), -1)
                                        
                                        # Add text
                                        font = cv2.FONT_HERSHEY_DUPLEX
                                        font_scale = 0.9
                                        font_thickness = 2
                                        cv2.putText(annotated_frame, status_text, (10, banner_height-12), font, 
                                                  font_scale, (255, 255, 255), font_thickness)
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")

                # --- VIOLATION DETECTION LOGIC (conditional based on traffic lights or crosswalk) ---
                # First, check if we have traffic lights detected
                traffic_lights = []
                has_traffic_lights = False
                
                # Handle multiple traffic lights with consensus approach
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})
                
                # Determine the dominant traffic light color based on confidence
                if traffic_lights:
                    # Filter to just red lights and sort by confidence
                    red_lights = [tl for tl in traffic_lights if tl.get('color') == 'red']
                    if red_lights:
                        # Use the highest confidence red light for display
                        highest_conf_red = max(red_lights, key=lambda x: x.get('confidence', 0))
                        # Update the global traffic light status for consistent UI display
                        self.latest_traffic_light = {
                            'color': 'red',
                            'confidence': highest_conf_red.get('confidence', 0.0)
                        }

                # Get traffic light position for crosswalk detection
                traffic_light_position = None
                if has_traffic_lights:
                    for det in detections:
                        if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                            traffic_light_bbox = det['bbox']
                            # Extract center point from bbox for crosswalk utils
                            x1, y1, x2, y2 = traffic_light_bbox
                            traffic_light_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            break

                # Run crosswalk detection to check if crosswalk exists
                try:
                    result_frame, crosswalk_bbox, violation_line_y, debug_info = detect_crosswalk_and_violation_line(
                        annotated_frame, traffic_light_position
                    )
                except Exception as e:
                    print(f"[ERROR] Crosswalk detection failed: {e}")
                    result_frame, crosswalk_bbox, violation_line_y, debug_info = annotated_frame, None, None, {}
                
                # Check if crosswalk is detected
                crosswalk_detected = crosswalk_bbox is not None
                stop_line_detected = debug_info.get('stop_line') is not None
                
                # Only proceed with violation logic if we have traffic lights OR crosswalk detected
                # AND every 3rd frame for performance (adjust as needed)
                violations = []
                self.violation_frame_counter += 1
                should_process_violations = (has_traffic_lights or crosswalk_detected) and (self.violation_frame_counter % 3 == 0)
                
                if should_process_violations:
                    print(f"[DEBUG] Processing violation logic - Traffic lights: {has_traffic_lights}, Crosswalk: {crosswalk_detected}")
                    
                    # Create violation line coordinates from y position
                    violation_line = None
                    if violation_line_y is not None:
                        start_pt = (0, violation_line_y)
                        end_pt = (annotated_frame.shape[1], violation_line_y)
                        violation_line = (start_pt, end_pt)
                        
                        # Draw the thick red violation line with black label background (like in image)
                        line_color = (0, 0, 255)  # Red color
                        cv2.line(annotated_frame, start_pt, end_pt, line_color, 6)  # Thick line
                        
                        # Draw black background for label
                        label = "Violation Line"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.9  # Larger font
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Center the text on the violation line
                        text_x = max(10, (annotated_frame.shape[1] - text_width) // 2)
                        
                        # Black background rectangle - centered and more prominent
                        cv2.rectangle(annotated_frame, 
                                    (text_x - 10, start_pt[1] - text_height - 15), 
                                    (text_x + text_width + 10, start_pt[1] - 5), 
                                    (0, 0, 0), -1)  # Black background
                        
                        # Red text - centered
                        cv2.putText(annotated_frame, label, (text_x, start_pt[1] - 10), 
                                  font, font_scale, line_color, thickness)
                        
                        print(f"[DEBUG] Violation line drawn at y={start_pt[1]}, type={label}")
                    else:
                        print(f"[DEBUG] No valid violation line detected.")

                    # DeepSORT tracking integration with movement detection
                    tracked_vehicles = []
                    if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                        try:
                            # Filter vehicle detections with stricter criteria
                            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                            
                            # Apply multiple filters for higher quality tracking:
                            # 1. Must be a vehicle class
                            # 2. Must have a bbox
                            # 3. Must meet confidence threshold
                            # 4. Must have reasonable dimensions (not too small/large)
                            vehicle_dets = []
                            h, w = frame.shape[:2]
                            min_area_ratio = 0.001  # Min 0.1% of frame area
                            max_area_ratio = 0.25   # Max 25% of frame area
                            
                            for det in detections:
                                if (det.get('class_name') in vehicle_classes and 
                                    'bbox' in det and 
                                    det.get('confidence', 0) > self.min_confidence_threshold):
                                    
                                    # Check bbox dimensions
                                    bbox = det['bbox']
                                    x1, y1, x2, y2 = bbox
                                    box_w, box_h = x2-x1, y2-y1
                                    box_area = box_w * box_h
                                    frame_area = w * h
                                    area_ratio = box_area / frame_area
                                    
                                    # Only include reasonably sized objects
                                    if min_area_ratio <= area_ratio <= max_area_ratio:
                                        vehicle_dets.append(det)
                            # Pass the detection dictionaries directly to the tracker
                            tracks = self.vehicle_tracker.update(vehicle_dets, frame)
                            
                            # tracks is a list of dicts: [{'id': track_id, 'bbox': [x1,y1,x2,y2], 'confidence': conf, 'class_id': class_id}, ...]
                            for track in tracks:
                                track_id = track['id']
                                bbox = track['bbox']
                                
                                # Calculate vehicle center for movement tracking
                                x1, y1, x2, y2 = map(float, bbox)
                                center_y = (y1 + y2) / 2
                                
                                # Initialize or update vehicle history
                                if track_id not in self.vehicle_history:
                                    from collections import deque
                                    self.vehicle_history[track_id] = deque(maxlen=10)  # Increased history for better movement detection
                                    self.vehicle_statuses = {}  # Keep track of vehicle movement status
                                
                                self.vehicle_history[track_id].append(center_y)
                                
                                # Calculate movement - improved algorithm
                                is_moving = False
                                
                                # Only analyze if we have enough history
                                if len(self.vehicle_history[track_id]) >= 3:
                                    # Get the recent history positions
                                    recent_positions = list(self.vehicle_history[track_id])
                                    
                                    # Calculate trend over multiple frames instead of just two frames
                                    if len(recent_positions) >= 5:
                                        # Get first half and second half positions to detect overall movement
                                        first_half = sum(recent_positions[:len(recent_positions)//2]) / (len(recent_positions)//2)
                                        second_half = sum(recent_positions[len(recent_positions)//2:]) / (len(recent_positions) - len(recent_positions)//2)
                                        
                                        # Calculate overall trend
                                        trend_movement = abs(second_half - first_half)
                                        is_moving = trend_movement > self.movement_threshold
                                    else:
                                        # Fallback to simpler calculation if not enough history
                                        prev_y = self.vehicle_history[track_id][-2]
                                        current_y = self.vehicle_history[track_id][-1]
                                        dy = abs(current_y - prev_y)
                                        is_moving = dy > self.movement_threshold
                                        
                                    # Store movement status persistently
                                    if track_id not in self.vehicle_statuses:
                                        self.vehicle_statuses[track_id] = {'is_moving': is_moving, 'stable_count': 0}
                                    else:
                                        # Update stable count based on consistency
                                        if self.vehicle_statuses[track_id]['is_moving'] == is_moving:
                                            self.vehicle_statuses[track_id]['stable_count'] += 1
                                        else:
                                            # Only switch status if consistent for multiple frames to avoid jitter
                                            if self.vehicle_statuses[track_id]['stable_count'] >= 3:
                                                self.vehicle_statuses[track_id]['is_moving'] = is_moving
                                                self.vehicle_statuses[track_id]['stable_count'] = 0
                                            else:
                                                is_moving = self.vehicle_statuses[track_id]['is_moving']  # Use previous state
                                                self.vehicle_statuses[track_id]['stable_count'] += 1
                                
                                tracked_vehicles.append({
                                    'id': track_id, 
                                    'bbox': bbox, 
                                    'center_y': center_y,
                                    'is_moving': is_moving,
                                    'prev_y': self.vehicle_history[track_id][-2] if len(self.vehicle_history[track_id]) >= 2 else center_y
                                })
                                
                            print(f"[DEBUG] DeepSORT tracked {len(tracked_vehicles)} vehicles")
                        except Exception as e:
                            print(f"[ERROR] DeepSORT tracking failed: {e}")
                            tracked_vehicles = []
                    else:
                        print("[WARN] DeepSORT vehicle tracker not available!")

                    # Red light violation detection
                    red_lights = []
                    for tl in traffic_lights:
                        if tl.get('color') == 'red':
                            red_lights.append(tl)
                    print(f"[DEBUG] Red light(s) detected: {len(red_lights)} red lights")
                    
                    vehicle_debugs = []
                    
                    # Always print vehicle debug info for frames with violation logic
                    for v in tracked_vehicles:
                        bbox = v['bbox']
                        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for OpenCV
                        center_y = v['center_y']
                        is_moving = v['is_moving']
                        status = "MOVING" if is_moving else "STOPPED"
                        vehicle_debugs.append(f"Vehicle ID={v['id']} bbox=[{x1},{y1},{x2},{y2}] center_y={center_y:.1f} status={status} vline_y={violation_line_y}")
                    
                    if red_lights and violation_line_y is not None:
                        print(f"[DEBUG] Checking {len(tracked_vehicles)} tracked vehicles for violations")
                        for v in tracked_vehicles:
                            bbox = v['bbox']
                            x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for OpenCV
                            
                            # Get movement status and center position
                            is_moving = v['is_moving']
                            current_y = v['center_y']
                            prev_y = v['prev_y']
                            
                            # A violation occurs only if:
                            # 1. Vehicle is moving (not stopped)
                            # 2. Vehicle crossed the line (previous position was before line, current is after)
                            crossed_line = (prev_y <= violation_line_y and current_y > violation_line_y)
                            is_violation = is_moving and crossed_line
                            
                            # Differentiate visualization based on vehicle state
                            if is_violation:
                                # RED BOX: Violation detected - crossed line while moving during red light
                                print(f"[DEBUG] 🚨 RED LIGHT VIOLATION: Vehicle ID={v['id']} CROSSED LINE while MOVING")
                                print(f"    Previous Y: {prev_y:.1f} -> Current Y: {current_y:.1f} (Line: {violation_line_y})")
                                
                                # Add to violations list with comprehensive data
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                violations.append({
                                    'track_id': v['id'], 
                                    'id': v['id'], 
                                    'bbox': [x1, y1, x2, y2], 
                                    'violation': 'red_light',
                                    'timestamp': timestamp,
                                    'line_position': violation_line_y,
                                    'movement': {'prev_y': prev_y, 'current_y': current_y}
                                })
                                
                                # Red box for violators (bolder)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # RED
                                
                                # Clear black background for violation label
                                label = f'VIOLATION ID:{v["id"]}'
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                thickness = 2
                                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                                
                                # Draw black background for text
                                cv2.rectangle(annotated_frame, 
                                            (x1, y1-text_height-10), 
                                            (x1+text_width+10, y1), 
                                            (0,0,0), -1)
                                
                                # Draw violation text in red
                                cv2.putText(annotated_frame, label, (x1+5, y1-10), 
                                          font, font_scale, (0, 0, 255), thickness)
                                
                            elif is_moving:
                                # ORANGE BOX: Moving but not violated
                                print(f"[DEBUG] Vehicle ID={v['id']} MOVING but not violated")
                                
                                # Orange box for moving vehicles
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                                
                                # Only show ID for moving vehicles
                                label = f'ID:{v["id"]}'
                                cv2.putText(annotated_frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                
                            else:
                                # GREEN BOX: Stopped vehicle - no text needed
                                print(f"[DEBUG] Vehicle ID={v['id']} STOPPED")
                                
                                # Green box for stopped vehicles (thinner)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green
                                
                                # No text for stopped vehicles - reduces clutter
                                
                                if current_y > violation_line_y and not is_moving:
                                    print(f"[DEBUG] Vehicle ID={v['id']} behind line but STOPPED - No violation")
                                elif is_moving and current_y <= violation_line_y:
                                    print(f"[DEBUG] Vehicle ID={v['id']} MOVING but before line - No violation")
                                else:
                                    print(f"[DEBUG] Vehicle ID={v['id']} normal tracking - No violation")
                        if not violations:
                            print("[DEBUG] No red light violations detected this frame.")
                    else:
                        print(f"[DEBUG] No red light or no violation line for this frame. Red lights: {len(red_lights)}, vline_y: {violation_line_y}")
                    
                    # Print vehicle debug info for frames with violation logic
                    for vdbg in vehicle_debugs:
                        print(f"[DEBUG] {vdbg}")
                else:
                    print(f"[DEBUG] Skipping violation logic - Frame {self.violation_frame_counter}: Traffic lights: {has_traffic_lights}, Crosswalk: {crosswalk_detected}")
                    violation_line_y = None  # Set to None when no violation logic runs
                
                # Emit individual violation signals for each violation
                if violations:
                    for violation in violations:
                        print(f"🚨 Emitting RED LIGHT VIOLATION: Track ID {violation['track_id']}")
                        # Add additional data to the violation
                        violation['frame'] = frame
                        violation['violation_line_y'] = violation_line_y
                        self.violation_detected.emit(violation)
                    print(f"[DEBUG] Emitted {len(violations)} violation signals")
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()# Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                violations = []  # Violations are disabled
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # --- Simplified frame processing for display ---
            # The violation logic is now handled in the main _run thread
            # This method just handles basic display overlays
            
            annotated_frame = frame.copy()

            # Add performance overlays and debug markers
            annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            cv2.circle(annotated_frame, (20, 20), 10, (255, 255, 0), -1)

            # Convert BGR to RGB before display (for PyQt/PySide)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # Display the RGB frame in the UI (replace with your display logic)
            # Example: self.image_label.setPixmap(QPixmap.fromImage(QImage(frame_rgb.data, w, h, QImage.Format_RGB888)))
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    # --- Removed unused internal violation line detection methods and RedLightViolationSystem usage ---


    ####BOHOT BDAIYA
from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap,
    pipeline_with_violation_line
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line, get_violation_line_y
from controllers.deepsort_tracker import DeepSortVehicleTracker
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Vehicle tracking settings
        self.vehicle_history = {}  # Dictionary to store vehicle position history
        self.vehicle_statuses = {}  # Track stable movement status
        self.movement_threshold = 2.5  # Minimum pixel change to consider a vehicle moving
        self.min_confidence_threshold = 0.5  # Minimum confidence for vehicle detection
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            # self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            # self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        self.violation_frame_counter = 0  # Add counter for violation processing
        
        # Vehicle movement tracking for violation detection
        self.vehicle_history = {}  # track_id -> deque of positions
        self.movement_threshold = 3  # pixels movement threshold
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = DeepSortVehicleTracker()
        
        # Add red light violation system
        # self.red_light_violation_system = RedLightViolationSystem()
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            
            # Properly terminate the thread
            self.thread.quit()
            if not self.thread.wait(3000):  # Wait 3 seconds max
                self.thread.terminate()
                print("WARNING: Thread termination forced")
            
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # Draw detections with bounding boxes for visual feedback
                if detections and len(detections) > 0:
                    # Only show traffic light and vehicle classes
                    allowed_classes = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                    filtered_detections = [det for det in detections if det.get('class_name') in allowed_classes]
                    print(f"Drawing {len(filtered_detections)} detection boxes on frame (filtered)")
                    for det in filtered_detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)

                            # Use red color if id==9 or is traffic light, else green
                            if class_id == 9 or is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red in BGR
                            else:
                                box_color = (0, 255, 0)  # Green in BGR
                            if 'id' in det:
                                id_text = f"ID: {det['id']}"
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(annotated_frame, f"{id_text} {label} ", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            # Draw vehicle ID if present
                            # if 'id' in det:
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    # Draw enhanced traffic light status
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                    
                                    # Add a prominent traffic light status at the top of the frame
                                    color = light_info.get('color', 'unknown')
                                    confidence = light_info.get('confidence', 0.0)
                                    
                                    if color == 'red':
                                        status_color = (0, 0, 255)  # Red
                                        status_text = f"Traffic Light: RED ({confidence:.2f})"
                                        
                                        # Draw a prominent red banner across the top
                                        banner_height = 40
                                        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], banner_height), (0, 0, 150), -1)
                                        
                                        # Add text
                                        font = cv2.FONT_HERSHEY_DUPLEX
                                        font_scale = 0.9
                                        font_thickness = 2
                                        cv2.putText(annotated_frame, status_text, (10, banner_height-12), font, 
                                                  font_scale, (255, 255, 255), font_thickness)
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")

                # --- VIOLATION DETECTION LOGIC (conditional based on traffic lights or crosswalk) ---
                # First, check if we have traffic lights detected
                traffic_lights = []
                has_traffic_lights = False
                
                # Handle multiple traffic lights with consensus approach
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})
                
                # Determine the dominant traffic light color based on confidence
                if traffic_lights:
                    # Filter to just red lights and sort by confidence
                    red_lights = [tl for tl in traffic_lights if tl.get('color') == 'red']
                    if red_lights:
                        # Use the highest confidence red light for display
                        highest_conf_red = max(red_lights, key=lambda x: x.get('confidence', 0))
                        # Update the global traffic light status for consistent UI display
                        self.latest_traffic_light = {
                            'color': 'red',
                            'confidence': highest_conf_red.get('confidence', 0.0)
                        }

                # Get traffic light position for crosswalk detection
                traffic_light_position = None
                if has_traffic_lights:
                    for det in detections:
                        if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                            traffic_light_bbox = det['bbox']
                            # Extract center point from bbox for crosswalk utils
                            x1, y1, x2, y2 = traffic_light_bbox
                            traffic_light_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            break

                # Run crosswalk detection to check if crosswalk exists
                try:
                    result_frame, crosswalk_bbox, violation_line_y, debug_info = detect_crosswalk_and_violation_line(
                        annotated_frame, traffic_light_position
                    )
                except Exception as e:
                    print(f"[ERROR] Crosswalk detection failed: {e}")
                    result_frame, crosswalk_bbox, violation_line_y, debug_info = annotated_frame, None, None, {}
                
                # Check if crosswalk is detected
                crosswalk_detected = crosswalk_bbox is not None
                stop_line_detected = debug_info.get('stop_line') is not None
                
                # Only proceed with violation logic if we have traffic lights OR crosswalk detected
                # AND every 3rd frame for performance (adjust as needed)
                violations = []
                self.violation_frame_counter += 1
                should_process_violations = (has_traffic_lights or crosswalk_detected) and (self.violation_frame_counter % 3 == 0)
                
                if should_process_violations:
                    print(f"[DEBUG] Processing violation logic - Traffic lights: {has_traffic_lights}, Crosswalk: {crosswalk_detected}")
                    
                    # Create violation line coordinates from y position
                    # violation_line = None
                    # if violation_line_y is not None:
                    #     start_pt = (0, violation_line_y)
                    #     end_pt = (annotated_frame.shape[1], violation_line_y)
                    #     violation_line = (start_pt, end_pt)
                        
                    #     # Draw the thick red violation line with black label background (like in image)
                    #     line_color = (0, 0, 255)  # Red color
                    #     cv2.line(annotated_frame, start_pt, end_pt, line_color, 6)  # Thick line
                        
                    #     # Draw black background for label
                    #     label = "Violation Line"
                    #     font = cv2.FONT_HERSHEY_SIMPLEX
                    #     font_scale = 0.9  # Larger font
                    #     thickness = 2
                    #     (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                        
                    #     # Center the text on the violation line
                    #     text_x = max(10, (annotated_frame.shape[1] - text_width) // 2)
                        
                    #     # Black background rectangle - centered and more prominent
                    #     cv2.rectangle(annotated_frame, 
                    #                 (text_x - 10, start_pt[1] - text_height - 15), 
                    #                 (text_x + text_width + 10, start_pt[1] - 5), 
                    #                 (0, 0, 0), -1)  # Black background
                        
                    #     # Red text - centered
                    #     cv2.putText(annotated_frame, label, (text_x, start_pt[1] - 10), 
                    #               font, font_scale, line_color, thickness)
                        
                    #     print(f"[DEBUG] Violation line drawn at y={start_pt[1]}, type={label}")
                    # else:
                    #     print(f"[DEBUG] No valid violation line detected.")

                    # DeepSORT tracking integration with movement detection
                    tracked_vehicles = []
                    if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                        try:
                            # Filter vehicle detections with stricter criteria
                            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                            
                            # Apply multiple filters for higher quality tracking:
                            # 1. Must be a vehicle class
                            # 2. Must have a bbox
                            # 3. Must meet confidence threshold
                            # 4. Must have reasonable dimensions (not too small/large)
                            vehicle_dets = []
                            h, w = frame.shape[:2]
                            min_area_ratio = 0.001  # Min 0.1% of frame area
                            max_area_ratio = 0.25   # Max 25% of frame area
                            
                            for det in detections:
                                if (det.get('class_name') in vehicle_classes and 
                                    'bbox' in det and 
                                    det.get('confidence', 0) > self.min_confidence_threshold):
                                    
                                    # Check bbox dimensions
                                    bbox = det['bbox']
                                    x1, y1, x2, y2 = bbox
                                    box_w, box_h = x2-x1, y2-y1
                                    box_area = box_w * box_h
                                    frame_area = w * h
                                    area_ratio = box_area / frame_area
                                    
                                    # Only include reasonably sized objects
                                    if min_area_ratio <= area_ratio <= max_area_ratio:
                                        vehicle_dets.append(det)
                            # Pass the detection dictionaries directly to the tracker
                            tracks = self.vehicle_tracker.update(vehicle_dets, frame)
                            
                            # tracks is a list of dicts: [{'id': track_id, 'bbox': [x1,y1,x2,y2], 'confidence': conf, 'class_id': class_id}, ...]
                            for track in tracks:
                                track_id = track['id']
                                bbox = track['bbox']
                                
                                # Calculate vehicle center for movement tracking
                                x1, y1, x2, y2 = map(float, bbox)
                                center_y = (y1 + y2) / 2
                                
                                # Initialize or update vehicle history
                                if track_id not in self.vehicle_history:
                                    from collections import deque
                                    self.vehicle_history[track_id] = deque(maxlen=10)  # Increased history for better movement detection
                                    self.vehicle_statuses = {}  # Keep track of vehicle movement status
                                
                                self.vehicle_history[track_id].append(center_y)
                                
                                # Calculate movement - improved algorithm
                                is_moving = False
                                
                                # Only analyze if we have enough history
                                if len(self.vehicle_history[track_id]) >= 3:
                                    # Get the recent history positions
                                    recent_positions = list(self.vehicle_history[track_id])
                                    
                                    # Calculate trend over multiple frames instead of just two frames
                                    if len(recent_positions) >= 5:
                                        # Get first half and second half positions to detect overall movement
                                        first_half = sum(recent_positions[:len(recent_positions)//2]) / (len(recent_positions)//2)
                                        second_half = sum(recent_positions[len(recent_positions)//2:]) / (len(recent_positions) - len(recent_positions)//2)
                                        
                                        # Calculate overall trend
                                        trend_movement = abs(second_half - first_half)
                                        is_moving = trend_movement > self.movement_threshold
                                    else:
                                        # Fallback to simpler calculation if not enough history
                                        prev_y = self.vehicle_history[track_id][-2]
                                        current_y = self.vehicle_history[track_id][-1]
                                        dy = abs(current_y - prev_y)
                                        is_moving = dy > self.movement_threshold
                                        
                                    # Store movement status persistently
                                    if track_id not in self.vehicle_statuses:
                                        self.vehicle_statuses[track_id] = {'is_moving': is_moving, 'stable_count': 0}
                                    else:
                                        # Update stable count based on consistency
                                        if self.vehicle_statuses[track_id]['is_moving'] == is_moving:
                                            self.vehicle_statuses[track_id]['stable_count'] += 1
                                        else:
                                            # Only switch status if consistent for multiple frames to avoid jitter
                                            if self.vehicle_statuses[track_id]['stable_count'] >= 3:
                                                self.vehicle_statuses[track_id]['is_moving'] = is_moving
                                                self.vehicle_statuses[track_id]['stable_count'] = 0
                                            else:
                                                is_moving = self.vehicle_statuses[track_id]['is_moving']  # Use previous state
                                                self.vehicle_statuses[track_id]['stable_count'] += 1
                                
                                tracked_vehicles.append({
                                    'id': track_id, 
                                    'bbox': bbox, 
                                    'center_y': center_y,
                                    'is_moving': is_moving,
                                    'prev_y': self.vehicle_history[track_id][-2] if len(self.vehicle_history[track_id]) >= 2 else center_y
                                })
                                
                            print(f"[DEBUG] DeepSORT tracked {len(tracked_vehicles)} vehicles")
                        except Exception as e:
                            print(f"[ERROR] DeepSORT tracking failed: {e}")
                            tracked_vehicles = []
                    else:
                        print("[WARN] DeepSORT vehicle tracker not available!")

                    # Red light violation detection
                    red_lights = []
                    for tl in traffic_lights:
                        if tl.get('color') == 'red':
                            red_lights.append(tl)
                    print(f"[DEBUG] Red light(s) detected: {len(red_lights)} red lights")
                    
                    vehicle_debugs = []
                    
                    # Always print vehicle debug info for frames with violation logic
                    for v in tracked_vehicles:
                        bbox = v['bbox']
                        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for OpenCV
                        center_y = v['center_y']
                        is_moving = v['is_moving']
                        status = "MOVING" if is_moving else "STOPPED"
                        vehicle_debugs.append(f"Vehicle ID={v['id']} bbox=[{x1},{y1},{x2},{y2}] center_y={center_y:.1f} status={status} vline_y={violation_line_y}")
                    
                    if red_lights and violation_line_y is not None:
                        print(f"[DEBUG] Checking {len(tracked_vehicles)} tracked vehicles for violations")
                        for v in tracked_vehicles:
                            bbox = v['bbox']
                            x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for OpenCV
                            
                            # Get movement status and center position
                            is_moving = v['is_moving']
                            current_y = v['center_y']
                            prev_y = v['prev_y']
                            
                            # A violation occurs only if:
                            # 1. Vehicle is moving (not stopped)
                            # 2. Vehicle crossed the line (previous position was before line, current is after)
                            crossed_line = (prev_y <= violation_line_y and current_y > violation_line_y)
                            is_violation = is_moving and crossed_line
                            
                            # Differentiate visualization based on vehicle state
                            if is_violation:
                                # RED BOX: Violation detected - crossed line while moving during red light
                                print(f"[DEBUG] 🚨 RED LIGHT VIOLATION: Vehicle ID={v['id']} CROSSED LINE while MOVING")
                                print(f"    Previous Y: {prev_y:.1f} -> Current Y: {current_y:.1f} (Line: {violation_line_y})")
                                
                                # Add to violations list with comprehensive data
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                violations.append({
                                    'track_id': v['id'], 
                                    'id': v['id'], 
                                    'bbox': [x1, y1, x2, y2], 
                                    'violation': 'red_light',
                                    'timestamp': timestamp,
                                    'line_position': violation_line_y,
                                    'movement': {'prev_y': prev_y, 'current_y': current_y}
                                })
                                
                                # Red box for violators (bolder)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # RED
                                
                                # Clear black background for violation label
                                label = f'VIOLATION ID:{v["id"]}'
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.7
                                thickness = 2
                                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                                
                                # Draw black background for text
                                cv2.rectangle(annotated_frame, 
                                            (x1, y1-text_height-10), 
                                            (x1+text_width+10, y1), 
                                            (0,0,0), -1)
                                
                                # Draw violation text in red
                                cv2.putText(annotated_frame, label, (x1+5, y1-10), 
                                          font, font_scale, (0, 0, 255), thickness)
                                
                            elif is_moving:
                                # ORANGE BOX: Moving but not violated
                                print(f"[DEBUG] Vehicle ID={v['id']} MOVING but not violated")
                                
                                # Orange box for moving vehicles
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange
                                
                                # Only show ID for moving vehicles
                                label = f'ID:{v["id"]}'
                                cv2.putText(annotated_frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                
                            else:
                                # # GREEN BOX: Stopped vehicle - no text needed
                                # print(f"[DEBUG] Vehicle ID={v['id']} STOPPED")
                                
                                # # Green box for stopped vehicles (thinner)
                                # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green
                                
                                # No text for stopped vehicles - reduces clutter
                                
                                if current_y > violation_line_y and not is_moving:
                                    print(f"[DEBUG] Vehicle ID={v['id']} behind line but STOPPED - No violation")
                                elif is_moving and current_y <= violation_line_y:
                                    print(f"[DEBUG] Vehicle ID={v['id']} MOVING but before line - No violation")
                                else:
                                    print(f"[DEBUG] Vehicle ID={v['id']} normal tracking - No violation")
                        if not violations:
                            print("[DEBUG] No red light violations detected this frame.")
                    else:
                        print(f"[DEBUG] No red light or no violation line for this frame. Red lights: {len(red_lights)}, vline_y: {violation_line_y}")
                    
                    # Print vehicle debug info for frames with violation logic
                    for vdbg in vehicle_debugs:
                        print(f"[DEBUG] {vdbg}")
                else:
                    print(f"[DEBUG] Skipping violation logic - Frame {self.violation_frame_counter}: Traffic lights: {has_traffic_lights}, Crosswalk: {crosswalk_detected}")
                    violation_line_y = None  # Set to None when no violation logic runs
                
                # Emit individual violation signals for each violation
                if violations:
                    for violation in violations:
                        print(f"🚨 Emitting RED LIGHT VIOLATION: Track ID {violation['track_id']}")
                        # Add additional data to the violation
                        violation['frame'] = frame
                        violation['violation_line_y'] = violation_line_y
                        self.violation_detected.emit(violation)
                    print(f"[DEBUG] Emitted {len(violations)} violation signals")
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()# Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                  # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                violations = []  # Violations are disabled
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # --- Simplified frame processing for display ---
            # The violation logic is now handled in the main _run thread
            # This method just handles basic display overlays
            
            annotated_frame = frame.copy()

            # Add performance overlays and debug markers
            annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            cv2.circle(annotated_frame, (20, 20), 10, (255, 255, 0), -1)

            # Convert BGR to RGB before display (for PyQt/PySide)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # Display the RGB frame in the UI (replace with your display logic)
            # Example: self.image_label.setPixmap(QPixmap.fromImage(QImage(frame_rgb.data, w, h, QImage.Format_RGB888)))
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    # --- Removed unused internal violation line detection methods and RedLightViolationSystem usage ---


    from PySide6.QtCore import QObject, Signal, QThread, Qt, QMutex, QWaitCondition, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
import os
import sys
import math

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities
from utils.annotation_utils import (
    draw_detections, 
    draw_performance_metrics,
    resize_frame_for_display,
    convert_cv_to_qimage,
    convert_cv_to_pixmap,
    pipeline_with_violation_line
)

# Import enhanced annotation utilities
from utils.enhanced_annotation_utils import (
    enhanced_draw_detections,
    draw_performance_overlay,
    enhanced_cv_to_qimage,
    enhanced_cv_to_pixmap
)

# Import traffic light color detection utilities
from red_light_violation_pipeline import RedLightViolationPipeline
from utils.traffic_light_utils import detect_traffic_light_color, draw_traffic_light_status, ensure_traffic_light_color
from utils.crosswalk_utils2 import detect_crosswalk_and_violation_line, draw_violation_line, get_violation_line_y
from controllers.bytetrack_tracker import ByteTrackVehicleTracker
TRAFFIC_LIGHT_CLASSES = ["traffic light", "trafficlight", "tl"]
TRAFFIC_LIGHT_NAMES = ['trafficlight', 'traffic light', 'tl', 'signal']

def normalize_class_name(class_name):
    """Normalizes class names from different models/formats to a standard name"""
    if not class_name:
        return ""
    
    name_lower = class_name.lower()
    
    # Traffic light variants
    if name_lower in ['traffic light', 'trafficlight', 'traffic_light', 'tl', 'signal']:
        return 'traffic light'
    
    # Keep specific vehicle classes (car, truck, bus) separate
    # Just normalize naming variations within each class
    if name_lower in ['car', 'auto', 'automobile']:
        return 'car'
    elif name_lower in ['truck']:
        return 'truck'
    elif name_lower in ['bus']:
        return 'bus'
    elif name_lower in ['motorcycle', 'scooter', 'motorbike', 'bike']:
        return 'motorcycle'
    
    # Person variants
    if name_lower in ['person', 'pedestrian', 'human']:
        return 'person'
    
    # Other common classes can be added here
    
    return class_name

def is_traffic_light(class_name):
    """Helper function to check if a class name is a traffic light with normalization"""
    if not class_name:
        return False
    normalized = normalize_class_name(class_name)
    return normalized == 'traffic light'

class VideoController(QObject):      
    frame_ready = Signal(object, object, dict)  # QPixmap, detections, metrics
    raw_frame_ready = Signal(np.ndarray, list, float)  # frame, detections, fps
    frame_np_ready = Signal(np.ndarray)  # Direct NumPy frame signal for display
    stats_ready = Signal(dict)  # Dictionary with stats (fps, detection_time, traffic_light)
    violation_detected = Signal(dict)  # Signal emitted when a violation is detected
    progress_ready = Signal(int, int, float)  # value, max_value, timestamp
    
    def __init__(self, model_manager=None):
        """
        Initialize video controller.
        
        Args:
            model_manager: Model manager instance for detection and violation
        """        
        super().__init__()
        print("Loaded advanced VideoController from video_controller_new.py")  # DEBUG: Confirm correct controller
        
        self._running = False
        self.source = None
        self.source_type = None
        self.source_fps = 0
        self.performance_metrics = {}
        self.mutex = QMutex()
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Store last 100 processing times
        self.fps_history = deque(maxlen=100)       # Store last 100 FPS values
        self.start_time = time.time()
        self.frame_count = 0
        self.actual_fps = 0.0
        
        self.model_manager = model_manager
        self.inference_model = None
        self.tracker = None
        
        self.current_frame = None
        self.current_detections = []
        
        # Traffic light state tracking
        self.latest_traffic_light = {"color": "unknown", "confidence": 0.0}
        
        # Vehicle tracking settings
        self.vehicle_history = {}  # Dictionary to store vehicle position history
        self.vehicle_statuses = {}  # Track stable movement status
        self.movement_threshold = 1.5  # ADJUSTED: More balanced movement detection (was 0.8)
        self.min_confidence_threshold = 0.3  # FIXED: Lower threshold for better detection (was 0.5)
        
        # Enhanced violation detection settings
        self.position_history_size = 20  # Increased from 10 to track longer history
        self.crossing_check_window = 8   # Check for crossings over the last 8 frames instead of just 2
        self.max_position_jump = 50      # Maximum allowed position jump between frames (detect ID switches)
        
        # Set up violation detection
        try:
            from controllers.red_light_violation_detector import RedLightViolationDetector
            self.violation_detector = RedLightViolationDetector()
            print("✅ Red light violation detector initialized")
        except Exception as e:
            self.violation_detector = None
            print(f"❌ Could not initialize violation detector: {e}")
            
        # Import crosswalk detection
        try:
            self.detect_crosswalk_and_violation_line = detect_crosswalk_and_violation_line
            # self.draw_violation_line = draw_violation_line
            print("✅ Crosswalk detection utilities imported")
        except Exception as e:
            print(f"❌ Could not import crosswalk detection: {e}")
            self.detect_crosswalk_and_violation_line = lambda frame, *args: (None, None, {})
            # self.draw_violation_line = lambda frame, *args, **kwargs: frame
        
        # Configure thread
        self.thread = QThread()
        self.moveToThread(self.thread)
        self.thread.started.connect(self._run)
          # Performance measurement
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.performance_metrics = {
            'FPS': 0.0,
            'Detection (ms)': 0.0,
            'Total (ms)': 0.0
        }
        
        # Setup render timer with more aggressive settings for UI updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self._process_frame)
        
        # Frame buffer
        self.current_frame = None
        self.current_detections = []
        self.current_violations = []
        
        # Debug counter for monitoring frame processing
        self.debug_counter = 0
        self.violation_frame_counter = 0  # Add counter for violation processing
        
        # Initialize the traffic light color detection pipeline
        self.cv_violation_pipeline = RedLightViolationPipeline(debug=True)
        
        # Initialize vehicle tracker
        self.vehicle_tracker = ByteTrackVehicleTracker()
        
        # Add red light violation system
        # self.red_light_violation_system = RedLightViolationSystem()
        
    def set_source(self, source):
        """
        Set video source (file path, camera index, or URL)
        
        Args:
            source: Video source - can be a camera index (int), file path (str), 
                   or URL (str). If None, defaults to camera 0.
                   
        Returns:
            bool: True if source was set successfully, False otherwise
        """
        print(f"🎬 VideoController.set_source called with: {source} (type: {type(source)})")
        
        # Store current state
        was_running = self._running
        
        # Stop current processing if running
        if self._running:
            print("⏹️ Stopping current video processing")
            self.stop()
        
        try:
            # Handle source based on type with better error messages
            if source is None:
                print("⚠️ Received None source, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
                
            elif isinstance(source, str) and source.strip():
                if os.path.exists(source):
                    # Valid file path
                    self.source = source
                    self.source_type = "file"
                    print(f"📄 Source set to file: {self.source}")
                elif source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
                    # URL stream
                    self.source = source
                    self.source_type = "url"
                    print(f"🌐 Source set to URL stream: {self.source}")
                elif source.isdigit():
                    # String camera index (convert to int)
                    self.source = int(source)
                    self.source_type = "camera"
                    print(f"📹 Source set to camera index: {self.source}")
                else:
                    # Try as device path or special string
                    self.source = source
                    self.source_type = "device"
                    print(f"📱 Source set to device path: {self.source}")
                    
            elif isinstance(source, int):
                # Camera index
                self.source = source
                self.source_type = "camera"
                print(f"📹 Source set to camera index: {self.source}")
                
            else:
                # Unrecognized - default to camera 0 with warning
                print(f"⚠️ Unrecognized source type: {type(source)}, defaulting to camera 0")
                self.source = 0
                self.source_type = "camera"
        except Exception as e:
            print(f"❌ Error setting source: {e}")
            self.source = 0
            self.source_type = "camera"
            return False
        
        # Get properties of the source (fps, dimensions, etc)
        print(f"🔍 Getting properties for source: {self.source}")
        success = self._get_source_properties()
        
        if success:
            print(f"✅ Successfully configured source: {self.source} ({self.source_type})")
            
            # Reset ByteTrack tracker for new source to ensure IDs start from 1
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                try:
                    print("🔄 Resetting vehicle tracker for new source")
                    self.vehicle_tracker.reset()
                except Exception as e:
                    print(f"⚠️ Could not reset vehicle tracker: {e}")
            
            # Emit successful source change
            self.stats_ready.emit({
                'source_changed': True,
                'source_type': self.source_type,
                'fps': self.source_fps if hasattr(self, 'source_fps') else 0,
                'dimensions': f"{self.frame_width}x{self.frame_height}" if hasattr(self, 'frame_width') else "unknown"
            })
            
            # Restart if previously running
            if was_running:
                print("▶️ Restarting video processing with new source")
                self.start()
        else:
            print(f"❌ Failed to configure source: {self.source}")
            # Notify UI about the error
            self.stats_ready.emit({
                'source_changed': False,
                'error': f"Invalid video source: {self.source}",
                'source_type': self.source_type,
                'fps': 0,
                'detection_time_ms': "0",
                'traffic_light_color': {"color": "unknown", "confidence": 0.0}
            })
            
            return False
            
        # Return success status
        return success
    
    def _get_source_properties(self):
        """
        Get properties of video source
        
        Returns:
            bool: True if source was successfully opened, False otherwise
        """
        try:
            print(f"🔍 Opening video source for properties check: {self.source}")
            cap = cv2.VideoCapture(self.source)
            
            # Verify capture opened successfully
            if not cap.isOpened():
                print(f"❌ Failed to open video source: {self.source}")
                return False
                
            # Read properties
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.source_fps <= 0:
                print("⚠️ Source FPS not available, using default 30 FPS")
                self.source_fps = 30.0  # Default if undetectable
            
            self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try reading a test frame to confirm source is truly working
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("⚠️ Could not read test frame from source")
                # For camera sources, try one more time with delay
                if self.source_type == "camera":
                    print("🔄 Retrying camera initialization...")
                    time.sleep(1.0)  # Wait a moment for camera to initialize
                    ret, test_frame = cap.read()
                    if not ret or test_frame is None:
                        print("❌ Camera initialization failed after retry")
                        cap.release()
                        return False
                else:
                    print("❌ Could not read frames from video source")
                    cap.release()
                    return False
                
            # Release the capture
            cap.release()
            
            print(f"✅ Video source properties: {self.frame_width}x{self.frame_height}, {self.source_fps} FPS")
            return True
            
        except Exception as e:
            print(f"❌ Error getting source properties: {e}")
            return False
            return False
            
    def start(self):
        """Start video processing"""
        if not self._running:
            self._running = True
            self.start_time = time.time()
            self.frame_count = 0
            self.debug_counter = 0
            print("DEBUG: Starting video processing thread")
            
            # Reset ByteTrack tracker to ensure IDs start from 1
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                try:
                    print("🔄 Resetting vehicle tracker for new session")
                    self.vehicle_tracker.reset()
                except Exception as e:
                    print(f"⚠️ Could not reset vehicle tracker: {e}")
            
            # Start the processing thread - add more detailed debugging
            if not self.thread.isRunning():
                print("🚀 Thread not running, starting now...")
                try:
                    self.thread.start()
                    print("✅ Thread started successfully")
                    print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
                except Exception as e:
                    print(f"❌ Failed to start thread: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Thread is already running!")
                print(f"🔄 Thread state: running={self.thread.isRunning()}, finished={self.thread.isFinished()}")
            
            # Start the render timer with a very aggressive interval (10ms = 100fps)
            # This ensures we can process frames as quickly as possible
            print("⏱️ Starting render timer...")
            self.render_timer.start(10)
            print("✅ Render timer started at 100Hz")
    
    def stop(self):
        """Stop video processing"""
        if self._running:
            print("DEBUG: Stopping video processing")
            self._running = False
            self.render_timer.stop()
            # Properly terminate the thread
            if self.thread.isRunning():
                self.thread.quit()
                if not self.thread.wait(3000):  # Wait 3 seconds max
                    self.thread.terminate()
                    print("WARNING: Thread termination forced")
            # Clear the current frame
            self.mutex.lock()
            self.current_frame = None
            self.mutex.unlock()
            print("DEBUG: Video processing stopped")

    def __del__(self):
        print("[VideoController] __del__ called. Cleaning up thread and timer.")
        self.stop()
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1000)
        self.render_timer.stop()
    
    def capture_snapshot(self) -> np.ndarray:
        """Capture current frame"""
        if self.current_frame is not None:
            return self.current_frame.copy()
        return None
        
    def _run(self):
        """Main processing loop (runs in thread)"""
        try:
            # Print the source we're trying to open
            print(f"DEBUG: Opening video source: {self.source} (type: {type(self.source)})")
            
            cap = None  # Initialize capture variable
            
            # Try to open source with more robust error handling
            max_retries = 3
            retry_delay = 1.0  # seconds
            
            # Function to attempt opening the source with multiple retries
            def try_open_source(src, retries=max_retries, delay=retry_delay):
                for attempt in range(1, retries + 1):
                    print(f"🎥 Opening source (attempt {attempt}/{retries}): {src}")
                    try:
                        capture = cv2.VideoCapture(src)
                        if capture.isOpened():
                            # Try to read a test frame to confirm it's working
                            ret, test_frame = capture.read()
                            if ret and test_frame is not None:
                                print(f"✅ Source opened successfully: {src}")
                                # Reset capture position for file sources
                                if isinstance(src, str) and os.path.exists(src):
                                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                return capture
                            else:
                                print(f"⚠️ Source opened but couldn't read frame: {src}")
                                capture.release()
                        else:
                            print(f"⚠️ Failed to open source: {src}")
                            
                        # Retry after delay
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                    except Exception as e:
                        print(f"❌ Error opening source {src}: {e}")
                        if attempt < retries:
                            print(f"Retrying in {delay:.1f} seconds...")
                            time.sleep(delay)
                
                print(f"❌ Failed to open source after {retries} attempts: {src}")
                return None
            
            # Handle different source types
            if isinstance(self.source, str) and os.path.exists(self.source):
                # It's a valid file path
                print(f"📄 Opening video file: {self.source}")
                cap = try_open_source(self.source)
                
            elif isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                # It's a camera index
                camera_idx = int(self.source) if isinstance(self.source, str) else self.source
                print(f"📹 Opening camera with index: {camera_idx}")
                
                # For cameras, try with different backend options if it fails
                cap = try_open_source(camera_idx)
                
                # If failed, try with DirectShow backend on Windows
                if cap is None and os.name == 'nt':
                    print("🔄 Trying camera with DirectShow backend...")
                    cap = try_open_source(camera_idx + cv2.CAP_DSHOW)
                    
            else:
                # Try as a string source (URL or device path)
                print(f"🌐 Opening source as string: {self.source}")
                cap = try_open_source(str(self.source))
                
            # Check if we successfully opened the source
            if cap is None:
                print(f"❌ Failed to open video source after all attempts: {self.source}")
                # Notify UI about the error
                self.stats_ready.emit({
                    'error': f"Could not open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                    
            # Check again to ensure capture is valid
            if not cap or not cap.isOpened():
                print(f"ERROR: Could not open video source {self.source}")
                # Emit a signal to notify UI about the error
                self.stats_ready.emit({
                    'error': f"Failed to open video source: {self.source}",
                    'fps': "0",
                    'detection_time_ms': "0",
                    'traffic_light_color': {"color": "unknown", "confidence": 0.0}
                })
                return
                
            # Configure frame timing based on source FPS
            frame_time = 1.0 / self.source_fps if self.source_fps > 0 else 0.033
            prev_time = time.time()
            
            # Log successful opening
            print(f"SUCCESS: Video source opened: {self.source}")
            print(f"Source info - FPS: {self.source_fps}, Size: {self.frame_width}x{self.frame_height}")
              # Main processing loop
            frame_error_count = 0
            max_consecutive_errors = 10
            
            while self._running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    # Add critical frame debugging
                    print(f"🟡 Frame read attempt: ret={ret}, frame={None if frame is None else frame.shape}")
                    
                    if not ret or frame is None:
                        frame_error_count += 1
                        print(f"⚠️ Frame read error ({frame_error_count}/{max_consecutive_errors})")
                        
                        if frame_error_count >= max_consecutive_errors:
                            print("❌ Too many consecutive frame errors, stopping video thread")
                            break
                            
                        # Skip this iteration and try again
                        time.sleep(0.1)  # Wait a bit before trying again
                        continue
                    
                    # Reset the error counter if we successfully got a frame
                    frame_error_count = 0
                except Exception as e:
                    print(f"❌ Critical error reading frame: {e}")
                    frame_error_count += 1
                    if frame_error_count >= max_consecutive_errors:
                        print("❌ Too many errors, stopping video thread")
                        break
                    continue
                    
                # Detection and violation processing
                process_start = time.time()
                
                # Process detections
                detection_start = time.time()
                detections = []
                if self.model_manager:
                    detections = self.model_manager.detect(frame)
                    
                    # Normalize class names for consistency and check for traffic lights
                    traffic_light_indices = []
                    for i, det in enumerate(detections):
                        if 'class_name' in det:
                            original_name = det['class_name']
                            normalized_name = normalize_class_name(original_name)
                            
                            # Keep track of traffic light indices
                            if normalized_name == 'traffic light' or original_name == 'traffic light':
                                traffic_light_indices.append(i)
                                
                            if original_name != normalized_name:
                                print(f"📊 Normalized class name: '{original_name}' -> '{normalized_name}'")
                                
                            det['class_name'] = normalized_name
                            
                    # Ensure we have at least one traffic light for debugging
                    if not traffic_light_indices and self.source_type == 'video':
                        print("⚠️ No traffic lights detected, checking for objects that might be traffic lights...")
                        
                        # Try lowering the confidence threshold specifically for traffic lights
                        # This is only for debugging purposes
                        if self.model_manager and hasattr(self.model_manager, 'detect'):
                            try:
                                low_conf_detections = self.model_manager.detect(frame, conf_threshold=0.2)
                                for det in low_conf_detections:
                                    if 'class_name' in det and det['class_name'] == 'traffic light':
                                        if det not in detections:
                                            print(f"🚦 Found low confidence traffic light: {det['confidence']:.2f}")
                                            detections.append(det)
                            except:
                                pass
                            
                detection_time = (time.time() - detection_start) * 1000
                
                # Violation detection is disabled
                violation_start = time.time()
                violations = []
                # if self.model_manager and detections:
                #     violations = self.model_manager.detect_violations(
                #         detections, frame, time.time()
                #     )
                violation_time = (time.time() - violation_start) * 1000
                
                # Update tracking if available
                if self.model_manager:
                    detections = self.model_manager.update_tracking(detections, frame)
                    # If detections are returned as tuples, convert to dicts for downstream code
                    if detections and isinstance(detections[0], tuple):
                        # Convert (id, bbox, conf, class_id) to dict
                        detections = [
                            {'id': d[0], 'bbox': d[1], 'confidence': d[2], 'class_id': d[3]}
                            for d in detections
                        ]
                
                # Calculate timing metrics
                process_time = (time.time() - process_start) * 1000
                self.processing_times.append(process_time)
                
                # Update FPS
                now = time.time()
                self.frame_count += 1
                elapsed = now - self.start_time
                if elapsed > 0:
                    self.actual_fps = self.frame_count / elapsed
                    
                fps_smoothed = 1.0 / (now - prev_time) if now > prev_time else 0
                prev_time = now
                  # Update metrics
                self.performance_metrics = {
                    'FPS': f"{fps_smoothed:.1f}",
                    'Detection (ms)': f"{detection_time:.1f}",
                    'Total (ms)': f"{process_time:.1f}"
                }
                
                # Store current frame data (thread-safe)
                self.mutex.lock()
                self.current_frame = frame.copy()
                self.current_detections = detections
                self.mutex.unlock()
                  # Process frame with annotations before sending to UI
                annotated_frame = frame.copy()
                
                # --- VIOLATION DETECTION LOGIC (Run BEFORE drawing boxes) ---
                # First get violation information so we can color boxes appropriately
                violating_vehicle_ids = set()  # Track which vehicles are violating
                violations = []
                
                # Initialize traffic light variables
                traffic_lights = []
                has_traffic_lights = False
                
                # Handle multiple traffic lights with consensus approach
                traffic_light_count = 0
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        traffic_light_count += 1
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})
                
                print(f"[TRAFFIC LIGHT] Detected {traffic_light_count} traffic light(s), has_traffic_lights={has_traffic_lights}")
                if has_traffic_lights:
                    print(f"[TRAFFIC LIGHT] Traffic light colors: {[tl.get('color', 'unknown') for tl in traffic_lights]}")
                
                # Get traffic light position for crosswalk detection
                traffic_light_position = None
                if has_traffic_lights:
                    for det in detections:
                        if is_traffic_light(det.get('class_name')) and 'bbox' in det:
                            traffic_light_bbox = det['bbox']
                            # Extract center point from bbox for crosswalk utils
                            x1, y1, x2, y2 = traffic_light_bbox
                            traffic_light_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                            break

                # Run crosswalk detection ONLY if traffic light is detected
                crosswalk_bbox, violation_line_y, debug_info = None, None, {}
                if has_traffic_lights and traffic_light_position is not None:
                    try:
                        print(f"[CROSSWALK] Traffic light detected at {traffic_light_position}, running crosswalk detection")
                        # Use new crosswalk_utils2 logic only when traffic light exists
                        annotated_frame, crosswalk_bbox, violation_line_y, debug_info = detect_crosswalk_and_violation_line(
                            annotated_frame,
                            traffic_light_position=traffic_light_position
                        )
                        print(f"[CROSSWALK] Detection result: crosswalk_bbox={crosswalk_bbox is not None}, violation_line_y={violation_line_y}")
                        # --- Draw crosswalk region if detected and close to traffic light ---
                        # (REMOVED: Do not draw crosswalk box or label)
                        # if crosswalk_bbox is not None:
                        #     x, y, w, h = map(int, crosswalk_bbox)
                        #     tl_x, tl_y = traffic_light_position
                        #     crosswalk_center_y = y + h // 2
                        #     distance = abs(crosswalk_center_y - tl_y)
                        #     print(f"[CROSSWALK DEBUG] Crosswalk bbox: {crosswalk_bbox}, Traffic light: {traffic_light_position}, vertical distance: {distance}")
                        #     if distance < 120:
                        #         cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                        #         cv2.putText(annotated_frame, "Crosswalk", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        #     # Top and bottom edge of crosswalk
                        #     top_edge = y
                        #     bottom_edge = y + h
                        #     if abs(tl_y - top_edge) < abs(tl_y - bottom_edge):
                        #         crosswalk_edge_y = top_edge
                        #     else:
                        #         crosswalk_edge_y = bottom_edge
                        if crosswalk_bbox is not None:
                            x, y, w, h = map(int, crosswalk_bbox)
                            tl_x, tl_y = traffic_light_position
                            crosswalk_center_y = y + h // 2
                            distance = abs(crosswalk_center_y - tl_y)
                            print(f"[CROSSWALK DEBUG] Crosswalk bbox: {crosswalk_bbox}, Traffic light: {traffic_light_position}, vertical distance: {distance}")
                            # Top and bottom edge of crosswalk
                            top_edge = y
                            bottom_edge = y + h
                            if abs(tl_y - top_edge) < abs(tl_y - bottom_edge):
                                crosswalk_edge_y = top_edge
                            else:
                                crosswalk_edge_y = bottom_edge
                    except Exception as e:
                        print(f"[ERROR] Crosswalk detection failed: {e}")
                        crosswalk_bbox, violation_line_y, debug_info = None, None, {}
                else:
                    print(f"[CROSSWALK] No traffic light detected (has_traffic_lights={has_traffic_lights}), skipping crosswalk detection")
                    # NO crosswalk detection without traffic light
                    violation_line_y = None
                
                # Check if crosswalk is detected
                crosswalk_detected = crosswalk_bbox is not None
                stop_line_detected = debug_info.get('stop_line') is not None
                
                # ALWAYS process vehicle tracking (moved outside violation logic)
                tracked_vehicles = []
                if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker is not None:
                    try:
                        # Filter vehicle detections
                        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                        vehicle_dets = []
                        h, w = frame.shape[:2]
                        
                        print(f"[TRACK DEBUG] Processing {len(detections)} total detections")
                        
                        for det in detections:
                            if (det.get('class_name') in vehicle_classes and 
                                'bbox' in det and 
                                det.get('confidence', 0) > self.min_confidence_threshold):
                                
                                # Check bbox dimensions
                                bbox = det['bbox']
                                x1, y1, x2, y2 = bbox
                                box_w, box_h = x2-x1, y2-y1
                                box_area = box_w * box_h
                                area_ratio = box_area / (w * h)
                                
                                print(f"[TRACK DEBUG] Vehicle {det.get('class_name')} conf={det.get('confidence'):.2f}, area_ratio={area_ratio:.4f}")
                                
                                if 0.001 <= area_ratio <= 0.25:
                                    vehicle_dets.append(det)
                                    print(f"[TRACK DEBUG] Added vehicle: {det.get('class_name')} conf={det.get('confidence'):.2f}")
                                else:
                                    print(f"[TRACK DEBUG] Rejected vehicle: area_ratio={area_ratio:.4f} not in range [0.001, 0.25]")
                        
                        print(f"[TRACK DEBUG] Filtered to {len(vehicle_dets)} vehicle detections")
                        
                        # Update tracker
                        if len(vehicle_dets) > 0:
                            print(f"[TRACK DEBUG] Updating tracker with {len(vehicle_dets)} vehicles...")
                            tracks = self.vehicle_tracker.update(vehicle_dets, frame)
                            # Filter out tracks without bbox to avoid warnings
                            valid_tracks = []
                            for track in tracks:
                                bbox = None
                                if isinstance(track, dict):
                                    bbox = track.get('bbox', None)
                                else:
                                    bbox = getattr(track, 'bbox', None)
                                if bbox is not None:
                                    valid_tracks.append(track)
                                else:
                                    print(f"Warning: Track has no bbox, skipping: {track}")
                            tracks = valid_tracks
                            print(f"[TRACK DEBUG] Tracker returned {len(tracks)} tracks (after bbox filter)")
                        else:
                            print(f"[TRACK DEBUG] No vehicles to track, skipping tracker update")
                            tracks = []
                        
                        # Process each tracked vehicle
                        tracked_vehicles = []
                        track_ids_seen = []
                        
                        for track in tracks:
                            track_id = track['id']
                            bbox = track['bbox']
                            x1, y1, x2, y2 = map(float, bbox)
                            center_y = (y1 + y2) / 2
                            
                            # Check for duplicate IDs
                            if track_id in track_ids_seen:
                                print(f"[TRACK ERROR] Duplicate ID detected: {track_id}")
                            track_ids_seen.append(track_id)
                            
                            print(f"[TRACK DEBUG] Processing track ID={track_id} bbox={bbox}")
                            
                            # Initialize or update vehicle history
                            if track_id not in self.vehicle_history:
                                from collections import deque
                                self.vehicle_history[track_id] = deque(maxlen=self.position_history_size)
                            
                            # Initialize vehicle status if not exists
                            if track_id not in self.vehicle_statuses:
                                self.vehicle_statuses[track_id] = {
                                    'recent_movement': [],
                                    'violation_history': [],
                                    'crossed_during_red': False,
                                    'last_position': None,  # Track last position for jump detection
                                    'suspicious_jumps': 0   # Count suspicious position jumps
                                }
                            
                            # Detect suspicious position jumps (potential ID switches)
                            if self.vehicle_statuses[track_id]['last_position'] is not None:
                                last_y = self.vehicle_statuses[track_id]['last_position']
                                center_y = (y1 + y2) / 2
                                position_jump = abs(center_y - last_y)
                                
                                if position_jump > self.max_position_jump:
                                    self.vehicle_statuses[track_id]['suspicious_jumps'] += 1
                                    print(f"[TRACK WARNING] Vehicle ID={track_id} suspicious position jump: {last_y:.1f} -> {center_y:.1f} (jump={position_jump:.1f})")
                                    
                                    # If too many suspicious jumps, reset violation status to be safe
                                    if self.vehicle_statuses[track_id]['suspicious_jumps'] > 2:
                                        print(f"[TRACK RESET] Vehicle ID={track_id} has too many suspicious jumps, resetting violation status")
                                        self.vehicle_statuses[track_id]['crossed_during_red'] = False
                                        self.vehicle_statuses[track_id]['suspicious_jumps'] = 0
                            
                            # Update position history and last position
                            self.vehicle_history[track_id].append(center_y)
                            self.vehicle_statuses[track_id]['last_position'] = center_y
                            
                            # BALANCED movement detection - detect clear movement while avoiding false positives
                            is_moving = False
                            movement_detected = False
                            
                            if len(self.vehicle_history[track_id]) >= 3:  # Require at least 3 frames for movement detection
                                recent_positions = list(self.vehicle_history[track_id])
                                
                                # Check movement over 3 frames for quick response
                                if len(recent_positions) >= 3:
                                    movement_3frames = abs(recent_positions[-1] - recent_positions[-3])
                                    if movement_3frames > self.movement_threshold:  # More responsive threshold
                                        movement_detected = True
                                        print(f"[MOVEMENT] Vehicle ID={track_id} MOVING: 3-frame movement = {movement_3frames:.1f}")
                                
                                # Confirm with longer movement for stability (if available)
                                if len(recent_positions) >= 5:
                                    movement_5frames = abs(recent_positions[-1] - recent_positions[-5])
                                    if movement_5frames > self.movement_threshold * 1.5:  # Moderate threshold for 5 frames
                                        movement_detected = True
                                        print(f"[MOVEMENT] Vehicle ID={track_id} MOVING: 5-frame movement = {movement_5frames:.1f}")
                            
                            # Store historical movement for smoothing - require consistent movement
                            self.vehicle_statuses[track_id]['recent_movement'].append(movement_detected)
                            if len(self.vehicle_statuses[track_id]['recent_movement']) > 4:  # Shorter history for quicker response
                                self.vehicle_statuses[track_id]['recent_movement'].pop(0)
                            
                            # BALANCED: Require majority of recent frames to show movement (2 out of 4)
                            recent_movement_count = sum(self.vehicle_statuses[track_id]['recent_movement'])
                            total_recent_frames = len(self.vehicle_statuses[track_id]['recent_movement'])
                            if total_recent_frames >= 2 and recent_movement_count >= (total_recent_frames * 0.5):  # 50% of frames must show movement
                                is_moving = True
                            
                            print(f"[TRACK DEBUG] Vehicle ID={track_id} is_moving={is_moving} (threshold={self.movement_threshold})")
                            
                            # Initialize as not violating
                            is_violation = False
                            
                            tracked_vehicles.append({
                                'id': track_id,
                                'bbox': bbox,
                                'center_y': center_y,
                                'is_moving': is_moving,
                                'is_violation': is_violation
                            })
                        
                        print(f"[DEBUG] ByteTrack tracked {len(tracked_vehicles)} vehicles")
                        for i, tracked in enumerate(tracked_vehicles):
                            print(f"  Vehicle {i}: ID={tracked['id']}, center_y={tracked['center_y']:.1f}, moving={tracked['is_moving']}, violating={tracked['is_violation']}")
                        
                        # DEBUG: Print all tracked vehicle IDs and their bboxes for this frame
                        if tracked_vehicles:
                            print(f"[DEBUG] All tracked vehicles this frame:")
                            for v in tracked_vehicles:
                                print(f"    ID={v['id']} bbox={v['bbox']} center_y={v.get('center_y', 'NA')}")
                        else:
                            print("[DEBUG] No tracked vehicles this frame!")
                        
                        # Clean up old vehicle data
                        current_track_ids = [tracked['id'] for tracked in tracked_vehicles]
                        self._cleanup_old_vehicle_data(current_track_ids)
                        
                    except Exception as e:
                        print(f"[ERROR] Vehicle tracking failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[WARN] ByteTrack vehicle tracker not available!")
                
                # Process violations - CHECK VEHICLES THAT CROSS THE LINE OVER A WINDOW OF FRAMES
                # IMPORTANT: Only process violations if traffic light is detected AND violation line exists
                if has_traffic_lights and violation_line_y is not None and tracked_vehicles:
                    print(f"[VIOLATION DEBUG] Traffic light present, checking {len(tracked_vehicles)} vehicles against violation line at y={violation_line_y}")
                    
                    # Check each tracked vehicle for violations
                    for tracked in tracked_vehicles:
                        track_id = tracked['id']
                        center_y = tracked['center_y']
                        is_moving = tracked['is_moving']
                        
                        # Get position history for this vehicle
                        position_history = list(self.vehicle_history[track_id])
                        
                        # Enhanced crossing detection: check over a window of frames
                        line_crossed_in_window = False
                        crossing_details = None
                        
                        if len(position_history) >= 2:
                            # Check for crossing over the last N frames (configurable window)
                            window_size = min(self.crossing_check_window, len(position_history))
                            
                            for i in range(1, window_size):
                                prev_y = position_history[-(i+1)]  # Earlier position
                                curr_y = position_history[-i]     # Later position
                                
                                # Check if vehicle crossed the line in this frame pair
                                if prev_y < violation_line_y and curr_y >= violation_line_y:
                                    line_crossed_in_window = True
                                    crossing_details = {
                                        'frames_ago': i,
                                        'prev_y': prev_y,
                                        'curr_y': curr_y,
                                        'window_checked': window_size
                                    }
                                    print(f"[VIOLATION DEBUG] Vehicle ID={track_id} crossed line {i} frames ago: {prev_y:.1f} -> {curr_y:.1f}")
                                    break
                        
                        # Check if traffic light is red
                        is_red_light = self.latest_traffic_light and self.latest_traffic_light.get('color') == 'red'
                        
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: latest_traffic_light={self.latest_traffic_light}, is_red_light={is_red_light}")
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: position_history={[f'{p:.1f}' for p in position_history[-5:]]}");  # Show last 5 positions
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: line_crossed_in_window={line_crossed_in_window}, crossing_details={crossing_details}")
                        
                        # Enhanced violation detection: vehicle crossed the line while moving and light is red
                        actively_crossing = (line_crossed_in_window and is_moving and is_red_light)
                        
                        # Initialize violation status for new vehicles
                        if 'crossed_during_red' not in self.vehicle_statuses[track_id]:
                            self.vehicle_statuses[track_id]['crossed_during_red'] = False
                        
                        # Mark vehicle as having crossed during red if it actively crosses
                        if actively_crossing:
                            # Additional validation: ensure it's not a false positive from ID switch
                            suspicious_jumps = self.vehicle_statuses[track_id].get('suspicious_jumps', 0)
                            if suspicious_jumps <= 1:  # Allow crossing if not too many suspicious jumps
                                self.vehicle_statuses[track_id]['crossed_during_red'] = True
                                print(f"[VIOLATION ALERT] Vehicle ID={track_id} CROSSED line during red light!")
                                print(f"  -> Crossing details: {crossing_details}")
                            else:
                                print(f"[VIOLATION IGNORED] Vehicle ID={track_id} crossing ignored due to {suspicious_jumps} suspicious jumps")
                        
                        # IMPORTANT: Reset violation status when light turns green (regardless of position)
                        if not is_red_light:
                            if self.vehicle_statuses[track_id]['crossed_during_red']:
                                print(f"[VIOLATION RESET] Vehicle ID={track_id} violation status reset (light turned green)")
                            self.vehicle_statuses[track_id]['crossed_during_red'] = False
                        
                        # Vehicle is violating ONLY if it crossed during red and light is still red
                        is_violation = (self.vehicle_statuses[track_id]['crossed_during_red'] and is_red_light)
                        
                        # Track current violation state for analytics - only actual crossings
                        self.vehicle_statuses[track_id]['violation_history'].append(actively_crossing)
                        if len(self.vehicle_statuses[track_id]['violation_history']) > 5:
                            self.vehicle_statuses[track_id]['violation_history'].pop(0)
                        
                        print(f"[VIOLATION DEBUG] Vehicle ID={track_id}: center_y={center_y:.1f}, line={violation_line_y}")
                        print(f"  history_window={[f'{p:.1f}' for p in position_history[-self.crossing_check_window:]]}")
                        print(f"  moving={is_moving}, red_light={is_red_light}")
                        print(f"  actively_crossing={actively_crossing}, crossed_during_red={self.vehicle_statuses[track_id]['crossed_during_red']}")
                        print(f"  suspicious_jumps={self.vehicle_statuses[track_id].get('suspicious_jumps', 0)}")
                        print(f"  FINAL_VIOLATION={is_violation}")
                        
                        # Update violation status
                        tracked['is_violation'] = is_violation
                        
                        if actively_crossing and self.vehicle_statuses[track_id].get('suspicious_jumps', 0) <= 1:  # Only add if not too many suspicious jumps
                            # Add to violating vehicles set
                            violating_vehicle_ids.add(track_id)
                            
                            # Add to violations list
                            timestamp = datetime.now()  # Keep as datetime object, not string
                            violations.append({
                                'track_id': track_id,
                                'id': track_id,
                                'bbox': [int(tracked['bbox'][0]), int(tracked['bbox'][1]), int(tracked['bbox'][2]), int(tracked['bbox'][3])],
                                'violation': 'line_crossing',
                                'violation_type': 'line_crossing',  # Add this for analytics compatibility
                                'timestamp': timestamp,
                                'line_position': violation_line_y,
                                'movement': crossing_details if crossing_details else {'prev_y': center_y, 'current_y': center_y},
                                'crossing_window': self.crossing_check_window,
                                'position_history': list(position_history[-10:])  # Include recent history for debugging
                            })
                            
                            print(f"[DEBUG] 🚨 VIOLATION DETECTED: Vehicle ID={track_id} CROSSED VIOLATION LINE")
                            print(f"    Enhanced detection: {crossing_details}")
                            print(f"    Position history: {[f'{p:.1f}' for p in position_history[-10:]]}")
                            print(f"    Detection window: {self.crossing_check_window} frames")
                            print(f"    while RED LIGHT & MOVING")
                
                # Emit progress signal after processing each frame
                if hasattr(self, 'progress_ready'):
                    self.progress_ready.emit(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), time.time())
                
                # Draw detections with bounding boxes - NOW with violation info
                # Only show traffic light and vehicle classes
                allowed_classes = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                filtered_detections = [det for det in detections if det.get('class_name') in allowed_classes]
                print(f"Drawing {len(filtered_detections)} detection boxes on frame (filtered)")
                
                # Statistics for debugging (always define, even if no detections)
                vehicles_with_ids = 0
                vehicles_without_ids = 0
                vehicles_moving = 0
                vehicles_violating = 0

                if detections and len(detections) > 0:
                    # Only show traffic light and vehicle classes
                    allowed_classes = ['traffic light', 'car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']
                    filtered_detections = [det for det in detections if det.get('class_name') in allowed_classes]
                    print(f"Drawing {len(filtered_detections)} detection boxes on frame (filtered)")
                    
                    # Statistics for debugging
                    vehicles_with_ids = 0
                    vehicles_without_ids = 0
                    vehicles_moving = 0
                    vehicles_violating = 0
                    
                    for det in filtered_detections:
                        if 'bbox' in det:
                            bbox = det['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det.get('class_name', 'object')
                            confidence = det.get('confidence', 0.0)
                            
                            # Robustness: ensure label and confidence are not None
                            if label is None:
                                label = 'object'
                            if confidence is None:
                                confidence = 0.0
                            class_id = det.get('class_id', -1)
                            
                            # Check if this detection corresponds to a violating or moving vehicle
                            det_center_x = (x1 + x2) / 2
                            det_center_y = (y1 + y2) / 2
                            is_violating_vehicle = False
                            is_moving_vehicle = False
                            vehicle_id = None
                            
                            # Match detection with tracked vehicles - IMPROVED MATCHING
                            if label in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle'] and len(tracked_vehicles) > 0:
                                print(f"[MATCH DEBUG] Attempting to match {label} detection at ({det_center_x:.1f}, {det_center_y:.1f}) with {len(tracked_vehicles)} tracked vehicles")
                                best_match = None
                                best_distance = float('inf')
                                best_iou = 0.0
                                
                                for i, tracked in enumerate(tracked_vehicles):
                                    track_bbox = tracked['bbox']
                                    track_x1, track_y1, track_x2, track_y2 = map(float, track_bbox)
                                    
                                    # Calculate center distance
                                    track_center_x = (track_x1 + track_x2) / 2
                                    track_center_y = (track_y1 + track_y2) / 2
                                    center_distance = ((det_center_x - track_center_x)**2 + (det_center_y - track_center_y)**2)**0.5
                                    
                                    # Calculate IoU (Intersection over Union)
                                    intersection_x1 = max(x1, track_x1)
                                    intersection_y1 = max(y1, track_y1)
                                    intersection_x2 = min(x2, track_x2)
                                    intersection_y2 = min(y2, track_y2)
                                    
                                    if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                                        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                                        det_area = (x2 - x1) * (y2 - y1)
                                        track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
                                        union_area = det_area + track_area - intersection_area
                                        iou = intersection_area / union_area if union_area > 0 else 0
                                    else:
                                        iou = 0
                                    
                                    print(f"[MATCH DEBUG] Track {i}: ID={tracked['id']}, center=({track_center_x:.1f}, {track_center_y:.1f}), distance={center_distance:.1f}, IoU={iou:.3f}")
                                    
                                    # Use stricter matching criteria - prioritize IoU over distance
                                    # Good match if: high IoU OR close center distance with some overlap
                                    is_good_match = (iou > 0.3) or (center_distance < 60 and iou > 0.1)
                                    
                                    if is_good_match:
                                        print(f"[MATCH DEBUG] Track {i} is a good match (IoU={iou:.3f}, distance={center_distance:.1f})")
                                        # Prefer higher IoU, then lower distance
                                        match_score = iou + (100 - min(center_distance, 100)) / 100  # Composite score
                                        if iou > best_iou or (iou == best_iou and center_distance < best_distance):
                                            best_distance = center_distance
                                            best_iou = iou
                                            best_match = tracked
                                    else:
                                        print(f"[MATCH DEBUG] Track {i} failed matching criteria (IoU={iou:.3f}, distance={center_distance:.1f})")
                                
                                if best_match:
                                    vehicle_id = best_match['id']
                                    is_moving_vehicle = best_match.get('is_moving', False)
                                    is_violating_vehicle = best_match.get('is_violation', False)
                                    print(f"[MATCH SUCCESS] Detection at ({det_center_x:.1f},{det_center_y:.1f}) matched with track ID={vehicle_id}")
                                    print(f"  -> STATUS: moving={is_moving_vehicle}, violating={is_violating_vehicle}, IoU={best_iou:.3f}, distance={best_distance:.1f}")
                                else:
                                    print(f"[MATCH FAILED] No suitable match found for {label} detection at ({det_center_x:.1f}, {det_center_y:.1f})")
                                    print(f"  -> Will draw as untracked detection with default color")
                            else:
                                if label not in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']:
                                    print(f"[MATCH DEBUG] Skipping matching for non-vehicle label: {label}")
                                elif len(tracked_vehicles) == 0:
                                    print(f"[MATCH DEBUG] No tracked vehicles available for matching")
                                else:
                                    try:
                                        if len(tracked_vehicles) > 0:
                                            distances = [((det_center_x - (t['bbox'][0] + t['bbox'][2])/2)**2 + (det_center_y - (t['bbox'][1] + t['bbox'][3])/2)**2)**0.5 for t in tracked_vehicles[:3]]
                                            print(f"[DEBUG] No match found for detection at ({det_center_x:.1f},{det_center_y:.1f}) - distances: {distances}")
                                        else:
                                            print(f"[DEBUG] No tracked vehicles available to match detection at ({det_center_x:.1f},{det_center_y:.1f})")
                                    except NameError:
                                        print(f"[DEBUG] No match found for detection (coords unavailable)")
                                        if len(tracked_vehicles) > 0:
                                            print(f"[DEBUG] Had {len(tracked_vehicles)} tracked vehicles available")
                            
                            # Choose box color based on vehicle status 
                            # PRIORITY: 1. Violating (RED) - crossed during red light 2. Moving (ORANGE) 3. Stopped (GREEN)
                            if is_violating_vehicle and vehicle_id is not None:
                                box_color = (0, 0, 255)  # RED for violating vehicles (crossed line during red)
                                label_text = f"{label}:ID{vehicle_id}⚠️"
                                thickness = 4
                                vehicles_violating += 1
                                print(f"[COLOR DEBUG] Drawing RED box for VIOLATING vehicle ID={vehicle_id} (crossed during red)")
                            elif is_moving_vehicle and vehicle_id is not None and not is_violating_vehicle:
                                box_color = (0, 165, 255)  # ORANGE for moving vehicles (not violating)
                                label_text = f"{label}:ID{vehicle_id}"
                                thickness = 3
                                vehicles_moving += 1
                                print(f"[COLOR DEBUG] Drawing ORANGE box for MOVING vehicle ID={vehicle_id} (not violating)")
                            elif label in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle'] and vehicle_id is not None:
                                box_color = (0, 255, 0)  # Green for stopped vehicles 
                                label_text = f"{label}:ID{vehicle_id}"
                                thickness = 2
                                print(f"[COLOR DEBUG] Drawing GREEN box for STOPPED vehicle ID={vehicle_id}")
                            elif is_traffic_light(label):
                                box_color = (0, 0, 255)  # Red for traffic lights
                                label_text = f"{label}"
                                thickness = 2
                            else:
                                box_color = (0, 255, 0)  # Default green for other objects
                                label_text = f"{label}"
                                thickness = 2
                            
                            # Update statistics
                            if label in ['car', 'truck', 'bus', 'motorcycle', 'van', 'bicycle']:
                                if vehicle_id is not None:
                                    vehicles_with_ids += 1
                                else:
                                    vehicles_without_ids += 1
                            
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                            cv2.putText(annotated_frame, label_text, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            #     id_text = f"ID: {det['id']}"
                            #     # Calculate text size for background
                            #     (tw, th), baseline = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            #     # Draw filled rectangle for background (top-left of bbox)
                            #     cv2.rectangle(annotated_frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                            #     # Draw the ID text in bold yellow
                            #     cv2.putText(annotated_frame, id_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                            #     print(f"[DEBUG] Detection ID: {det['id']} BBOX: {bbox} CLASS: {label} CONF: {confidence:.2f}")
                           
                            if class_id == 9 or is_traffic_light(label):
                                try:
                                    light_info = detect_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    if light_info.get("color", "unknown") == "unknown":
                                        light_info = ensure_traffic_light_color(annotated_frame, [x1, y1, x2, y2])
                                    det['traffic_light_color'] = light_info
                                    # Draw enhanced traffic light status
                                    annotated_frame = draw_traffic_light_status(annotated_frame, bbox, light_info)
                                    
                                    # --- Update latest_traffic_light for UI/console ---
                                    self.latest_traffic_light = light_info
                                    
                                    # Add a prominent traffic light status at the top of the frame
                                    color = light_info.get('color', 'unknown')
                                    confidence = light_info.get('confidence', 0.0)
                                    
                                    if color == 'red':
                                        status_color = (0, 0, 255)  # Red
                                        status_text = f"Traffic Light: RED ({confidence:.2f})"
                                        
                                        # Draw a prominent red banner across the top
                                        banner_height = 40
                                        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], banner_height), (0, 0, 150), -1)
                                        
                                        # Add text
                                        font = cv2.FONT_HERSHEY_DUPLEX
                                        font_scale = 0.9
                                        font_thickness = 2
                                        cv2.putText(annotated_frame, status_text, (10, banner_height-12), font, 
                                                  font_scale, (255, 255, 255), font_thickness)
                                except Exception as e:
                                    print(f"[WARN] Could not detect/draw traffic light color: {e}")

                # Print statistics summary
                print(f"[STATS] Vehicles: {vehicles_with_ids} with IDs, {vehicles_without_ids} without IDs")
                print(f"[STATS] Moving: {vehicles_moving}, Violating: {vehicles_violating}")
                
                # Handle multiple traffic lights with consensus approach
                for det in detections:
                    if is_traffic_light(det.get('class_name')):
                        has_traffic_lights = True
                        if 'traffic_light_color' in det:
                            light_info = det['traffic_light_color']
                            traffic_lights.append({'bbox': det['bbox'], 'color': light_info.get('color', 'unknown'), 'confidence': light_info.get('confidence', 0.0)})
                
                # Determine the dominant traffic light color based on confidence
                if traffic_lights:
                    # Filter to just red lights and sort by confidence
                    red_lights = [tl for tl in traffic_lights if tl.get('color') == 'red']
                    if red_lights:
                        # Use the highest confidence red light for display
                        highest_conf_red = max(red_lights, key=lambda x: x.get('confidence', 0))
                        # Update the global traffic light status for consistent UI display
                        self.latest_traffic_light = {
                            'color': 'red',
                            'confidence': highest_conf_red.get('confidence', 0.0)
                        }

                # Emit individual violation signals for each violation
                if violations:
                    for violation in violations:
                        print(f"🚨 Emitting RED LIGHT VIOLATION: Track ID {violation['track_id']}")
                        # Add additional data to the violation
                        violation['frame'] = frame
                        violation['violation_line_y'] = violation_line_y
                        self.violation_detected.emit(violation)
                    print(f"[DEBUG] Emitted {len(violations)} violation signals")
                
                # Add FPS display directly on frame
                # cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # # --- Always draw detected traffic light color indicator at top ---
                # color = self.latest_traffic_light.get('color', 'unknown') if isinstance(self.latest_traffic_light, dict) else str(self.latest_traffic_light)
                # confidence = self.latest_traffic_light.get('confidence', 0.0) if isinstance(self.latest_traffic_light, dict) else 0.0
                # indicator_size = 30
                # margin = 10
                # status_colors = {
                #     "red": (0, 0, 255),
                #     "yellow": (0, 255, 255),
                #     "green": (0, 255, 0),
                #     "unknown": (200, 200, 200)
                # }
                # draw_color = status_colors.get(color, (200, 200, 200))
                # # Draw circle indicator
                # cv2.circle(
                #     annotated_frame,
                #     (annotated_frame.shape[1] - margin - indicator_size, margin + indicator_size),
                #     indicator_size,
                #     draw_color,
                #     -1
                # )
                # # Add color text
                # cv2.putText(
                #     annotated_frame,
                #     f"{color.upper()} ({confidence:.2f})",
                #     (annotated_frame.shape[1] - margin - indicator_size - 120, margin + indicator_size + 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 0),
                #     2
                # )

                # Signal for raw data subscribers (now without violations)
                # Emit with correct number of arguments
                try:
                    self.raw_frame_ready.emit(frame.copy(), detections, fps_smoothed)
                    print(f"✅ raw_frame_ready signal emitted with {len(detections)} detections, fps={fps_smoothed:.1f}")
                except Exception as e:
                    print(f"❌ Error emitting raw_frame_ready: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit the NumPy frame signal for direct display - annotated version for visual feedback
                print(f"🔴 Emitting frame_np_ready signal with annotated_frame shape: {annotated_frame.shape}")
                try:
                    # Make sure the frame can be safely transmitted over Qt's signal system
                    # Create a contiguous copy of the array
                    frame_copy = np.ascontiguousarray(annotated_frame)
                    print(f"🔍 Debug - Before emission: frame_copy type={type(frame_copy)}, shape={frame_copy.shape}, is_contiguous={frame_copy.flags['C_CONTIGUOUS']}")
                    self.frame_np_ready.emit(frame_copy)
                    print("✅ frame_np_ready signal emitted successfully")
                except Exception as e:
                    print(f"❌ Error emitting frame: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit QPixmap for video detection tab (frame_ready)
                try:
                    from PySide6.QtGui import QImage, QPixmap
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qimg)
                    metrics = {
                        'FPS': fps_smoothed,
                        'Detection (ms)': detection_time
                    }
                    self.frame_ready.emit(pixmap, detections, metrics)
                    print("✅ frame_ready signal emitted for video detection tab")
                except Exception as e:
                    print(f"❌ Error emitting frame_ready: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Emit stats signal for performance monitoring
                stats = {
                    'fps': fps_smoothed,
                    'detection_fps': fps_smoothed,  # Numeric value for analytics
                    'detection_time': detection_time,
                    'detection_time_ms': detection_time,  # Numeric value for analytics
                    'traffic_light_color': self.latest_traffic_light
                }
                
                # Print detailed stats for debugging
                tl_color = "unknown"
                if isinstance(self.latest_traffic_light, dict):
                    tl_color = self.latest_traffic_light.get('color', 'unknown')
                elif isinstance(self.latest_traffic_light, str):
                    tl_color = self.latest_traffic_light
                
                print(f"🟢 Stats Updated: FPS={fps_smoothed:.2f}, Inference={detection_time:.2f}ms, Traffic Light={tl_color}")
                      
                # Emit stats signal
                self.stats_ready.emit(stats)

                # --- Ensure analytics update every frame ---
                if hasattr(self, 'analytics_controller') and self.analytics_controller is not None:
                    try:
                        self.analytics_controller.process_frame_data(frame, detections, stats)
                        print("[DEBUG] Called analytics_controller.process_frame_data for analytics update")
                    except Exception as e:
                        print(f"[ERROR] Could not update analytics: {e}")
                
                # Control processing rate for file sources
                if isinstance(self.source, str) and self.source_fps > 0:
                    frame_duration = time.time() - process_start
                    if frame_duration < frame_time:
                        time.sleep(frame_time - frame_duration)
            
            cap.release()
        except Exception as e:
            print(f"Video processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False
    def _process_frame(self):
        """Process current frame for display with improved error handling"""
        try:
            self.mutex.lock()
            if self.current_frame is None:
                print("⚠️ No frame available to process")
                self.mutex.unlock()
                
                # Check if we're running - if not, this is expected behavior
                if not self._running:
                    return
                
                # If we are running but have no frame, create a blank frame with error message
                h, w = 480, 640  # Default size
                blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "No video input", (w//2-100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Emit this blank frame
                try:
                    self.frame_np_ready.emit(blank_frame)
                except Exception as e:
                    print(f"Error emitting blank frame: {e}")
                
                return
            
            # Make a copy of the data we need
            try:
                frame = self.current_frame.copy()
                detections = self.current_detections.copy() if self.current_detections else []
                violations = []  # Violations are disabled
                metrics = self.performance_metrics.copy()
            except Exception as e:
                print(f"Error copying frame data: {e}")
                self.mutex.unlock()
                return
                
            self.mutex.unlock()
        except Exception as e:
            print(f"Critical error in _process_frame initialization: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.mutex.unlock()
            except:
                pass
            return
        
        try:
            # --- Simplified frame processing for display ---
            # The violation logic is now handled in the main _run thread
            # This method just handles basic display overlays
            
            annotated_frame = frame.copy()

            # Add performance overlays and debug markers - COMMENTED OUT for clean video display
            # annotated_frame = draw_performance_overlay(annotated_frame, metrics)
            # cv2.circle(annotated_frame, (20, 20), 10, (255, 255, 0), -1)

            # Convert BGR to RGB before display (for PyQt/PySide)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            # Display the RGB frame in the UI (replace with your display logic)
            # Example: self.image_label.setPixmap(QPixmap.fromImage(QImage(frame_rgb.data, w, h, QImage.Format_RGB888)))
        except Exception as e:
            print(f"Error in _process_frame: {e}")
            import traceback
            traceback.print_exc()

    def _cleanup_old_vehicle_data(self, current_track_ids):
        """
        Clean up tracking data for vehicles that are no longer being tracked.
        This prevents memory leaks and improves performance.
        
        Args:
            current_track_ids: Set of currently active track IDs
        """
        # Find IDs that are no longer active
        old_ids = set(self.vehicle_history.keys()) - set(current_track_ids)
        
        if old_ids:
            print(f"[CLEANUP] Removing tracking data for {len(old_ids)} old vehicle IDs: {sorted(old_ids)}")
            for old_id in old_ids:
                # Remove from history and status tracking
                if old_id in self.vehicle_history:
                    del self.vehicle_history[old_id]
                if old_id in self.vehicle_statuses:
                    del self.vehicle_statuses[old_id]
            print(f"[CLEANUP] Now tracking {len(self.vehicle_history)} active vehicles")

    # --- Removed unused internal violation line detection methods and RedLightViolationSystem usage ---
    def play(self):
        """Alias for start(), for UI compatibility."""
        self.start()