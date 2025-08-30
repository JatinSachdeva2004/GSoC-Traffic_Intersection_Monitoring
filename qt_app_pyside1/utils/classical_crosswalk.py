import cv2
import numpy as np
import math
from sklearn import linear_model

def lineCalc(vx, vy, x0, y0):
    scale = 10
    x1 = x0 + scale * vx
    y1 = y0 + scale * vy
    m = (y1 - y0) / (x1 - x0)
    b = y1 - m * x1
    return m, b

def lineIntersect(m1, b1, m2, b2):
    a_1 = -m1
    b_1 = 1
    c_1 = b1
    a_2 = -m2
    b_2 = 1
    c_2 = b2
    d = a_1 * b_2 - a_2 * b_1
    dx = c_1 * b_2 - c_2 * b_1
    dy = a_1 * c_2 - a_2 * c_1
    intersectionX = dx / d
    intersectionY = dy / d
    return intersectionX, intersectionY

def detect_crosswalk(frame):
    '''Detects crosswalk/zebra lines robustly for various camera angles using adaptive thresholding and Hough Line Transform.'''
    import cv2
    import numpy as np
    img = frame.copy()
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding for lighting invariance
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 7)
    # Morphology to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 30, 3))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Hough Line Transform to find lines
    lines = cv2.HoughLinesP(morphed, 1, np.pi / 180, threshold=80, minLineLength=W // 10, maxLineGap=20)
    crosswalk_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter for nearly horizontal lines (crosswalk stripes)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -20 < angle < 20:  # adjust as needed for your camera
                crosswalk_lines.append((x1, y1, x2, y2))
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # If no crosswalk lines found, return
    if not crosswalk_lines:
        return None, [], img
    # Use the lowest (max y) line as the violation line
    violation_line_y = max([max(y1, y2) for (x1, y1, x2, y2) in crosswalk_lines])
    cv2.line(img, (0, violation_line_y), (W, violation_line_y), (0, 0, 255), 2)
    return violation_line_y, crosswalk_lines, img

if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    vp, medians, vis = detect_crosswalk(img)
    cv2.imshow("Crosswalk Detection", vis)
    cv2.waitKey(0)
