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
    '''Detects crosswalk/zebra lines and vanishing point in a BGR frame.'''
    H, W = frame.shape[:2]
    radius = 250
    bw_width = 170
    lower = np.array([170, 170, 170])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lower, upper)
    erodeSize = int(H / 30)
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize, 1))
    erode = cv2.erode(mask, erodeStructure, (-1, -1))
    contours, _ = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bxbyLeftArray, bxbyRightArray = [], []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw > bw_width:
            cv2.line(frame, (bx, by), (bx + bw, by), (0, 255, 0), 2)
            bxbyLeftArray.append([bx, by])
            bxbyRightArray.append([bx + bw, by])
            cv2.circle(frame, (int(bx), int(by)), 5, (0, 250, 250), 2)
            cv2.circle(frame, (int(bx + bw), int(by)), 5, (250, 250, 0), 2)
    if len(bxbyLeftArray) < 2 or len(bxbyRightArray) < 2:
        return None, None, frame
    medianL = np.median(bxbyLeftArray, axis=0)
    medianR = np.median(bxbyRightArray, axis=0)
    boundedLeft = [i for i in bxbyLeftArray if ((medianL[0] - i[0]) ** 2 + (medianL[1] - i[1]) ** 2) < radius ** 2]
    boundedRight = [i for i in bxbyRightArray if ((medianR[0] - i[0]) ** 2 + (medianR[1] - i[1]) ** 2) < radius ** 2]
    if len(boundedLeft) < 2 or len(boundedRight) < 2:
        return None, None, frame
    bxLeft = np.asarray([pt[0] for pt in boundedLeft]).reshape(-1, 1)
    byLeft = np.asarray([pt[1] for pt in boundedLeft])
    bxRight = np.asarray([pt[0] for pt in boundedRight]).reshape(-1, 1)
    byRight = np.asarray([pt[1] for pt in boundedRight])
    modelL = linear_model.RANSACRegressor().fit(bxLeft, byLeft)
    modelR = linear_model.RANSACRegressor().fit(bxRight, byRight)
    vx, vy, x0, y0 = cv2.fitLine(np.array(boundedLeft), cv2.DIST_L2, 0, 0.01, 0.01)
    vx_R, vy_R, x0_R, y0_R = cv2.fitLine(np.array(boundedRight), cv2.DIST_L2, 0, 0.01, 0.01)
    m_L, b_L = lineCalc(vx, vy, x0, y0)
    m_R, b_R = lineCalc(vx_R, vy_R, x0_R, y0_R)
    intersectionX, intersectionY = lineIntersect(m_R, b_R, m_L, b_L)
    m = radius * 10
    if intersectionY < H / 2:
        cv2.circle(frame, (int(intersectionX), int(intersectionY)), 10, (0, 0, 255), 15)
        cv2.line(frame, (int(x0 - m * vx), int(y0 - m * vy)), (int(x0 + m * vx), int(y0 + m * vy)), (255, 0, 0), 3)
        cv2.line(frame, (int(x0_R - m * vx_R), int(y0_R - m * vy_R)), (int(x0_R + m * vx_R), int(y0_R + m * vy_R)), (255, 0, 0), 3)
    return (int(intersectionX), int(intersectionY)), [list(medianL) + list(medianR)], frame
