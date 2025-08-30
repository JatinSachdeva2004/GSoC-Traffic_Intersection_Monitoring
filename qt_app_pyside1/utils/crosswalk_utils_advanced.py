print("🔧 [CROSSWALK_UTILS_ADVANCED] Advanced crosswalk detection with CLAHE, HSV, Sobel, and hierarchical clustering LOADED")
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

# Try to import scipy for hierarchical clustering, fallback to simple grouping
try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
    print("[CROSSWALK_ADVANCED] Scipy available - using hierarchical clustering")
except ImportError:
    SCIPY_AVAILABLE = False
    print("[CROSSWALK_ADVANCED] Scipy not available - using simple grouping")

def detect_crosswalk_and_violation_line(frame: np.ndarray, traffic_light_position: Optional[Tuple[int, int]] = None, perspective_M: Optional[np.ndarray] = None):
    """
    Advanced crosswalk detection using CLAHE, HSV, Sobel, and hierarchical clustering.
    
    Args:
        frame: BGR image frame from video feed
        traffic_light_position: Optional (x, y) of traffic light in frame
        perspective_M: Optional 3x3 homography matrix for bird's eye view normalization
    
    Returns:
        result_frame: frame with overlays (for visualization)
        crosswalk_bbox: (x, y, w, h) or None if fallback used
        violation_line_y: int (y position for violation check)
        debug_info: dict (for visualization/debugging)
    """
    print(f"[CROSSWALK_ADVANCED] Starting advanced detection. Traffic light: {traffic_light_position}")
    
    debug_info = {}
    orig_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # 1️⃣ PERSPECTIVE NORMALIZATION (Bird's Eye View)
    if perspective_M is not None:
        frame = cv2.warpPerspective(frame, perspective_M, (w, h))
        debug_info['perspective_warped'] = True
        print("[CROSSWALK_ADVANCED] Applied perspective warping")
    else:
        debug_info['perspective_warped'] = False
    
    # 2️⃣ ADVANCED PREPROCESSING
    
    # CLAHE-enhanced grayscale for shadow and low-light handling
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    debug_info['clahe_applied'] = True
    
    # HSV + V channel for bright white detection robust to hue variations
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask_white = cv2.inRange(v, 180, 255)
    debug_info['hsv_white_ratio'] = np.sum(mask_white > 0) / (h * w)
    
    # Blend mask with adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    combined = cv2.bitwise_and(thresh, mask_white)
    
    # 3️⃣ EDGE DETECTION WITH SOBEL HORIZONTAL EMPHASIS
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    
    # Combine Sobel with white mask for better stripe detection
    sobel_combined = cv2.bitwise_and(sobelx, mask_white)
    
    # 4️⃣ MORPHOLOGICAL ENHANCEMENT
    
    # Horizontal kernel to connect broken stripes
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    
    # Vertical kernel to remove vertical noise
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_v, iterations=1)
    
    # Additional processing with Sobel results
    sobel_morph = cv2.morphologyEx(sobel_combined, cv2.MORPH_CLOSE, kernel_h, iterations=1)
    
    # Combine both approaches
    final_mask = cv2.bitwise_or(morph, sobel_morph)
    
    # 5️⃣ CONTOUR EXTRACTION WITH ADVANCED FILTERING
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Focus on lower ROI where crosswalks typically are
    roi_y_start = int(h * 0.4)
    zebra_stripes = []
    
    for cnt in contours:
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        
        # Skip if in upper part of frame
        if y < roi_y_start:
            continue
        
        # Advanced filtering criteria
        aspect_ratio = w_rect / max(h_rect, 1)
        area = w_rect * h_rect
        normalized_width = w_rect / w
        
        # 1. Aspect Ratio: Wide and short
        if aspect_ratio < 2.0:
            continue
            
        # 2. Area: Covers meaningful width
        min_area = 200
        max_area = 0.25 * h * w
        if not (min_area < area < max_area):
            continue
            
        # 3. Coverage: Should cover significant width
        if normalized_width < 0.05:  # At least 5% of frame width
            continue
            
        # 4. Parallelism: Check if stripe is roughly horizontal
        if len(cnt) >= 5:
            [vx, vy, cx, cy] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.degrees(np.arctan2(vy, vx))
            if not (abs(angle) < 15 or abs(angle) > 165):
                continue
        
        zebra_stripes.append({
            'contour': cnt,
            'bbox': (x, y, w_rect, h_rect),
            'center': (x + w_rect//2, y + h_rect//2),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'normalized_width': normalized_width
        })
    
    print(f"[CROSSWALK_ADVANCED] Found {len(zebra_stripes)} potential zebra stripes")
    
    # 6️⃣ STRIPE GROUPING (Hierarchical Clustering or Simple Grouping)
    crosswalk_bbox = None
    violation_line_y = None
    
    if len(zebra_stripes) >= 2:
        if SCIPY_AVAILABLE:
            # Use hierarchical clustering
            clusters = perform_hierarchical_clustering(zebra_stripes, h)
        else:
            # Use simple distance-based grouping
            clusters = perform_simple_grouping(zebra_stripes, h)
        
        # 7️⃣ ADVANCED SCORING FOR CROSSWALK IDENTIFICATION
        scored_clusters = []
        
        for cluster_id, stripes in clusters.items():
            if len(stripes) < 2:  # Need at least 2 stripes
                continue
                
            score = calculate_crosswalk_score(stripes, w, h)
            scored_clusters.append((score, stripes, cluster_id))
        
        debug_info['clusters_found'] = len(clusters)
        debug_info['scored_clusters'] = len(scored_clusters)
        
        if scored_clusters:
            # Select best cluster
            scored_clusters.sort(reverse=True, key=lambda x: x[0])
            best_score, best_stripes, best_cluster_id = scored_clusters[0]
            
            print(f"[CROSSWALK_ADVANCED] Best cluster score: {best_score:.3f} with {len(best_stripes)} stripes")
            
            if best_score > 0.3:  # Threshold for valid crosswalk
                # Calculate crosswalk bounding box
                all_bboxes = [s['bbox'] for s in best_stripes]
                xs = [bbox[0] for bbox in all_bboxes] + [bbox[0] + bbox[2] for bbox in all_bboxes]
                ys = [bbox[1] for bbox in all_bboxes] + [bbox[1] + bbox[3] for bbox in all_bboxes]
                
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                crosswalk_bbox = (x1, y1, x2 - x1, y2 - y1)
                
                # Place violation line before crosswalk
                violation_line_y = y1 - 20
                
                debug_info['crosswalk_detected'] = True
                debug_info['crosswalk_score'] = best_score
                debug_info['crosswalk_bbox'] = crosswalk_bbox
                debug_info['best_stripes'] = best_stripes
                
                print(f"[CROSSWALK_ADVANCED] CROSSWALK DETECTED at bbox: {crosswalk_bbox}")
                print(f"[CROSSWALK_ADVANCED] Violation line at y={violation_line_y}")
    
    # 8️⃣ FALLBACK: ENHANCED STOP-LINE DETECTION
    if crosswalk_bbox is None:
        print("[CROSSWALK_ADVANCED] No crosswalk found, using stop-line detection fallback")
        violation_line_y = detect_stop_line_fallback(frame, traffic_light_position, h, w, debug_info)
    
    # 9️⃣ TRAFFIC LIGHT ALIGNMENT (if provided)
    if traffic_light_position and violation_line_y:
        violation_line_y = align_violation_line_to_traffic_light(
            violation_line_y, traffic_light_position, crosswalk_bbox, h
        )
        debug_info['traffic_light_aligned'] = True
    
    # 🔟 VISUALIZATION
    result_frame = orig_frame.copy()
    if violation_line_y is not None:
        result_frame = draw_violation_line(result_frame, violation_line_y, 
                                         color=(0, 0, 255), thickness=8, 
                                         style='solid', label='VIOLATION LINE')
    
    # Draw crosswalk bbox if detected
    if crosswalk_bbox:
        x, y, w_box, h_box = crosswalk_bbox
        cv2.rectangle(result_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 3)
        cv2.putText(result_frame, 'CROSSWALK', (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_frame, crosswalk_bbox, violation_line_y, debug_info

def draw_violation_line(frame: np.ndarray, y: int, color=(0, 0, 255), thickness=8, style='solid', label='Violation Line'):
    """
    Draws a thick, optionally dashed, labeled violation line at the given y-coordinate.
    Args:
        frame: BGR image
        y: y-coordinate for the line
        color: BGR color tuple
        thickness: line thickness
        style: 'solid' or 'dashed'
        label: Optional label to draw above the line
    Returns:
        frame with line overlay
    """
    import cv2
    h, w = frame.shape[:2]
    x1, x2 = 0, w
    overlay = frame.copy()
    if style == 'dashed':
        dash_len = 30
        gap = 20
        for x in range(x1, x2, dash_len + gap):
            x_end = min(x + dash_len, x2)
            cv2.line(overlay, (x, y), (x_end, y), color, thickness, lineType=cv2.LINE_AA)
    else:
        cv2.line(overlay, (x1, y), (x2, y), color, thickness, lineType=cv2.LINE_AA)
    # Blend for semi-transparency
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    # Draw label
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(label, font, 0.8, 2)
        text_x = max(10, (w - text_size[0]) // 2)
        text_y = max(0, y - 12)
        cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0,0,0), -1)
        cv2.putText(frame, label, (text_x, text_y), font, 0.8, color, 2, cv2.LINE_AA)
    return frame

def get_violation_line_y(frame, traffic_light_bbox=None, crosswalk_bbox=None):
    """
    Returns the y-coordinate of the violation line using the following priority:
    1. Crosswalk bbox (most accurate)
    2. Stop line detection via image processing (CV)
    3. Traffic light bbox heuristic
    4. Fallback (default)
    """
    height, width = frame.shape[:2]
    # 1. Crosswalk bbox
    if crosswalk_bbox is not None and len(crosswalk_bbox) == 4:
        return int(crosswalk_bbox[1]) - 15
    # 2. Stop line detection (CV)
    roi_height = int(height * 0.4)
    roi_y = height - roi_height
    roi = frame[roi_y:height, 0:width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, -2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stop_line_candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)
        normalized_width = w / width
        if (aspect_ratio > 5 and normalized_width > 0.3 and h < 15 and y > roi_height * 0.5):
            abs_y = y + roi_y
            stop_line_candidates.append((abs_y, w))
    if stop_line_candidates:
        stop_line_candidates.sort(key=lambda x: x[1], reverse=True)
        return stop_line_candidates[0][0]
    # 3. Traffic light bbox heuristic
    if traffic_light_bbox is not None and len(traffic_light_bbox) == 4:
        traffic_light_bottom = traffic_light_bbox[3]
        traffic_light_height = traffic_light_bbox[3] - traffic_light_bbox[1]
        estimated_distance = min(5 * traffic_light_height, height * 0.3)
        return min(int(traffic_light_bottom + estimated_distance), height - 20)
  
def calculate_crosswalk_score(stripes: List[Dict], frame_width: int, frame_height: int) -> float:
    """
    Advanced scoring function for crosswalk validation using multiple criteria.
    
    Args:
        stripes: List of stripe dictionaries with bbox, area, etc.
        frame_width: Width of the frame
        frame_height: Height of the frame
    
    Returns:
        score: Float between 0-1, higher is better
    """
    if len(stripes) < 2:
        return 0.0
    
    # Extract metrics
    heights = [s['bbox'][3] for s in stripes]
    widths = [s['bbox'][2] for s in stripes]
    y_centers = [s['center'][1] for s in stripes]
    x_centers = [s['center'][0] for s in stripes]
    areas = [s['area'] for s in stripes]
    
    # 1. Stripe Count Score (more stripes = more confident)
    count_score = min(len(stripes) / 5.0, 1.0)  # Optimal around 5 stripes
    
    # 2. Height Consistency Score
    if len(heights) > 1:
        height_std = np.std(heights)
        height_mean = np.mean(heights)
        height_score = max(0, 1.0 - (height_std / (height_mean + 1e-6)))
    else:
        height_score = 0.5
    
    # 3. Horizontal Alignment Score (y-coordinates should be similar)
    if len(y_centers) > 1:
        y_std = np.std(y_centers)
        y_tolerance = frame_height * 0.05  # 5% of frame height
        y_score = max(0, 1.0 - (y_std / y_tolerance))
    else:
        y_score = 0.5
    
    # 4. Regular Spacing Score
    if len(stripes) >= 3:
        x_sorted = sorted(x_centers)
        gaps = [x_sorted[i+1] - x_sorted[i] for i in range(len(x_sorted)-1)]
        gap_mean = np.mean(gaps)
        gap_std = np.std(gaps)
        spacing_score = max(0, 1.0 - (gap_std / (gap_mean + 1e-6)))
    else:
        spacing_score = 0.3
    
    # 5. Coverage Score (should span reasonable width)
    total_width = max(x_centers) - min(x_centers)
    coverage_ratio = total_width / frame_width
    coverage_score = min(coverage_ratio / 0.3, 1.0)  # Target 30% coverage
    
    # 6. Area Consistency Score
    if len(areas) > 1:
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        area_score = max(0, 1.0 - (area_std / (area_mean + 1e-6)))
    else:
        area_score = 0.5
    
    # 7. Aspect Ratio Consistency Score
    aspect_ratios = [s['aspect_ratio'] for s in stripes]
    if len(aspect_ratios) > 1:
        aspect_std = np.std(aspect_ratios)
        aspect_mean = np.mean(aspect_ratios)
        aspect_score = max(0, 1.0 - (aspect_std / (aspect_mean + 1e-6)))
    else:
        aspect_score = 0.5
    
    # Weighted final score
    weights = {
        'count': 0.2,
        'height': 0.15,
        'alignment': 0.2,
        'spacing': 0.15,
        'coverage': 0.15,
        'area': 0.075,
        'aspect': 0.075
    }
    
    final_score = (
        weights['count'] * count_score +
        weights['height'] * height_score +
        weights['alignment'] * y_score +
        weights['spacing'] * spacing_score +
        weights['coverage'] * coverage_score +
        weights['area'] * area_score +
        weights['aspect'] * aspect_score
    )
    
    return final_score

def detect_stop_line_fallback(frame: np.ndarray, traffic_light_position: Optional[Tuple[int, int]], 
                             frame_height: int, frame_width: int, debug_info: Dict) -> Optional[int]:
    """
    Enhanced stop-line detection using Canny + HoughLinesP with improved filtering.
    
    Args:
        frame: Input frame
        traffic_light_position: Optional traffic light position
        frame_height: Height of frame
        frame_width: Width of frame
        debug_info: Debug information dictionary
    
    Returns:
        violation_line_y: Y-coordinate of violation line or None
    """
    print("[CROSSWALK_ADVANCED] Running stop-line detection fallback")
    
    # Convert to grayscale and apply CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Focus on lower ROI where stop lines typically are
    roi_height = int(frame_height * 0.6)  # Lower 60% of frame
    roi_y = frame_height - roi_height
    roi_gray = gray[roi_y:frame_height, :]
    
    # Enhanced edge detection
    edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)
    
    # Morphological operations to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Detect horizontal lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                           threshold=40, minLineLength=int(frame_width * 0.2), maxLineGap=20)
    
    stop_line_candidates = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Convert back to full frame coordinates
            y1 += roi_y
            y2 += roi_y
            
            # Calculate line properties
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            line_center_y = (y1 + y2) // 2
            
            # Filter for horizontal lines
            if (abs(angle) < 10 or abs(angle) > 170) and line_length > frame_width * 0.15:
                stop_line_candidates.append({
                    'line': (x1, y1, x2, y2),
                    'center_y': line_center_y,
                    'length': line_length,
                    'angle': angle
                })
    
    debug_info['stop_line_candidates'] = len(stop_line_candidates)
    
    if stop_line_candidates:
        # Score and select best stop line
        best_line = None
        
        if traffic_light_position:
            tx, ty = traffic_light_position
            # Find line that's appropriately positioned relative to traffic light
            valid_candidates = [
                candidate for candidate in stop_line_candidates 
                if candidate['center_y'] > ty + 30  # Below traffic light
            ]
            
            if valid_candidates:
                # Select line closest to expected distance from traffic light
                expected_distance = frame_height * 0.3  # 30% of frame height
                target_y = ty + expected_distance
                
                best_candidate = min(valid_candidates, 
                                   key=lambda c: abs(c['center_y'] - target_y))
                best_line = best_candidate['line']
            else:
                # Fallback to longest line
                best_candidate = max(stop_line_candidates, key=lambda c: c['length'])
                best_line = best_candidate['line']
        else:
            # Select the bottom-most line with good length
            best_candidate = max(stop_line_candidates, 
                               key=lambda c: c['center_y'] + c['length'] * 0.1)
            best_line = best_candidate['line']
        
        if best_line:
            x1, y1, x2, y2 = best_line
            violation_line_y = min(y1, y2) - 15  # 15 pixels before stop line
            debug_info['stop_line_used'] = best_line
            print(f"[CROSSWALK_ADVANCED] Stop line detected, violation line at y={violation_line_y}")
            return violation_line_y
    
    # Final fallback - use heuristic based on frame and traffic light
    if traffic_light_position:
        tx, ty = traffic_light_position
        fallback_y = int(ty + frame_height * 0.25)  # 25% below traffic light
    else:
        fallback_y = int(frame_height * 0.75)  # 75% down the frame
    
    debug_info['fallback_used'] = True
    print(f"[CROSSWALK_ADVANCED] Using fallback violation line at y={fallback_y}")
    return fallback_y

def align_violation_line_to_traffic_light(violation_line_y: int, traffic_light_position: Tuple[int, int], 
                                         crosswalk_bbox: Optional[Tuple], frame_height: int) -> int:
    """
    Align violation line dynamically based on traffic light position.
    
    Args:
        violation_line_y: Current violation line y-coordinate
        traffic_light_position: (x, y) of traffic light
        crosswalk_bbox: Crosswalk bounding box if detected
        frame_height: Height of frame
    
    Returns:
        adjusted_violation_line_y: Adjusted y-coordinate
    """
    tx, ty = traffic_light_position
    
    # Calculate expected distance from traffic light to violation line
    if crosswalk_bbox:
        # If crosswalk detected, maintain current position but validate
        expected_distance = frame_height * 0.2  # 20% of frame height
        actual_distance = violation_line_y - ty
        
        # If too close or too far, adjust slightly
        if actual_distance < expected_distance * 0.5:
            violation_line_y = int(ty + expected_distance * 0.7)
        elif actual_distance > expected_distance * 2:
            violation_line_y = int(ty + expected_distance * 1.3)
    else:
        # For stop lines, use standard distance
        standard_distance = frame_height * 0.25  # 25% of frame height
        violation_line_y = int(ty + standard_distance)
    
    # Ensure violation line is within frame bounds
    violation_line_y = max(20, min(violation_line_y, frame_height - 20))
    
    print(f"[CROSSWALK_ADVANCED] Traffic light aligned violation line at y={violation_line_y}")
    return violation_line_y

def perform_hierarchical_clustering(zebra_stripes: List[Dict], frame_height: int) -> Dict:
    """
    Perform hierarchical clustering on zebra stripes using scipy.
    
    Args:
        zebra_stripes: List of stripe dictionaries
        frame_height: Height of frame for distance threshold
    
    Returns:
        clusters: Dictionary of cluster_id -> list of stripes
    """
    # Extract y-coordinates for clustering
    y_coords = np.array([stripe['center'][1] for stripe in zebra_stripes]).reshape(-1, 1)
    
    if len(y_coords) <= 1:
        return {1: zebra_stripes}
    
    # Perform hierarchical clustering
    distances = pdist(y_coords, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    
    # Get clusters (max distance threshold)
    max_distance = frame_height * 0.08  # 8% of frame height
    cluster_labels = fcluster(linkage_matrix, max_distance, criterion='distance')
    
    # Group stripes by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(zebra_stripes[i])
    
    return clusters

def perform_simple_grouping(zebra_stripes: List[Dict], frame_height: int) -> Dict:
    """
    Perform simple distance-based grouping when scipy is not available.
    
    Args:
        zebra_stripes: List of stripe dictionaries
        frame_height: Height of frame for distance threshold
    
    Returns:
        clusters: Dictionary of cluster_id -> list of stripes
    """
    if not zebra_stripes:
        return {}
    
    # Sort stripes by y-coordinate
    sorted_stripes = sorted(zebra_stripes, key=lambda s: s['center'][1])
    
    clusters = {}
    cluster_id = 1
    y_tolerance = frame_height * 0.08  # 8% of frame height
    
    current_cluster = [sorted_stripes[0]]
    
    for i in range(1, len(sorted_stripes)):
        current_stripe = sorted_stripes[i]
        prev_stripe = sorted_stripes[i-1]
        
        y_diff = abs(current_stripe['center'][1] - prev_stripe['center'][1])
        
        if y_diff <= y_tolerance:
            # Add to current cluster
            current_cluster.append(current_stripe)
        else:
            # Start new cluster
            if len(current_cluster) >= 2:  # Only keep clusters with 2+ stripes
                clusters[cluster_id] = current_cluster
                cluster_id += 1
            current_cluster = [current_stripe]
    
    # Don't forget the last cluster
    if len(current_cluster) >= 2:
        clusters[cluster_id] = current_cluster
    
    return clusters

# Example usage:
# bbox, vline, dbg = detect_crosswalk_and_violation_line(frame, (tl_x, tl_y), perspective_M)