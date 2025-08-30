import cv2
import numpy as np

def findNonZero(rgb_image):
    rows, cols, _ = rgb_image.shape
    counter = 0
    for row in range(rows):
        for col in range(cols):
            pixel = rgb_image[row, col]
            if sum(pixel) != 0:
                counter += 1
    return counter

def red_green_yellow(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:,:,1])
    area = rgb_image.shape[0] * rgb_image.shape[1]
    avg_saturation = sum_saturation / area
    sat_low = int(avg_saturation * 1.3)
    val_low = 140
    lower_green = np.array([70,sat_low,val_low])
    upper_green = np.array([100,255,255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    lower_yellow = np.array([10,sat_low,val_low])
    upper_yellow = np.array([60,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_red = np.array([150,sat_low,val_low])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    sum_green = findNonZero(cv2.bitwise_and(rgb_image, rgb_image, mask=green_mask))
    sum_yellow = findNonZero(cv2.bitwise_and(rgb_image, rgb_image, mask=yellow_mask))
    sum_red = findNonZero(cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask))
    if sum_red >= sum_yellow and sum_red >= sum_green:
        return "red"
    if sum_yellow >= sum_green:
        return "yellow"
    return "green"

def detect_traffic_light_color(frame, bbox):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return red_green_yellow(roi_rgb)

if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    bbox = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    color = detect_traffic_light_color(img, bbox)
    print("Detected color:", color)
