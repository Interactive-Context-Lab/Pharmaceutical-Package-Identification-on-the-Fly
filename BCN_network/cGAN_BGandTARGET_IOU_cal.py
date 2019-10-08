# coding=utf-8

import cv2
import numpy as np
import math
import time
from itertools import combinations, permutations

import os


def GetAreaOfPolyGon(points):
    area = 0
    if (len(points) < 3):
        raise Exception("must have 3 points")
    p1 = points[0]
    for i in range(1, len(points) - 1):
        p2 = points[i]
        p3 = points[i + 1]
        vecp1p2 = [p2[0] - p1[0], p2[1] - p1[1]]
        vecp2p3 = [p3[0] - p2[0], p3[1] - p2[1]]
        vecMult = vecp1p2[0] * vecp2p3[1] - vecp1p2[1] * vecp2p3[0]
        sign = 0
        if (vecMult > 0):
            sign = 1
        elif (vecMult < 0):
            sign = -1
        triArea = GetAreaOfTriangle(p1, p2, p3) * sign
        area += triArea
    return abs(area)


def GetAreaOfTriangle(p1, p2, p3):
    area = 0
    p1p2 = GetLineLength(p1, p2)
    p2p3 = GetLineLength(p2, p3)
    p3p1 = GetLineLength(p3, p1)
    s = (p1p2 + p2p3 + p3p1) / 2
    area = s * (s - p1p2) * (s - p2p3) * (s - p3p1)
    area = math.sqrt(area)
    return area


def GetLineLength(p1, p2):
    length = math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2)
    length = math.sqrt(length)
    return length


def iou_cal(imgA, imgB):
    imgA_tmp = imgA.copy()

    imgB_tmp = imgB.copy()

    imgA_tmp = cv2.cvtColor(imgA_tmp, cv2.COLOR_BGR2GRAY)
    imgB_tmp = cv2.cvtColor(imgB_tmp, cv2.COLOR_BGR2GRAY)

    retA, imgA_tmp = cv2.threshold(imgA_tmp, 127, 255, cv2.THRESH_BINARY)
    retB, imgB_tmp = cv2.threshold(imgB_tmp, 127, 255, cv2.THRESH_BINARY)

    _, contoursA, hierarchyA = cv2.findContours(imgA_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0  # 找出最大面積
    index = 0
    for cntA in range(len(contoursA)):
        area = cv2.contourArea(contoursA[cntA])
        if area > max_area:
            max_area = area
            index = cntA
    if max_area < 50 or len(contoursA[index]) < 3:
        imgA_tmp3 = np.zeros((256, 256), np.uint8)
    else:
        blank_img = np.zeros(imgA_tmp.shape, np.uint8)
        blank_img1 = np.zeros(imgA_tmp.shape, np.uint8)
        hull2 = cv2.convexHull(contoursA[index])
        cv2.drawContours(blank_img1, [hull2], 0, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

        a = cv2.cvtColor(blank_img1, cv2.COLOR_GRAY2BGR)

        epsilon = 0.01 * cv2.arcLength(hull2, True)
        approx = cv2.approxPolyDP(hull2, epsilon, True)
        cv2.drawContours(a, approx, -1, (0, 0, 255), 3)

        if len(approx) > 4:  # 偵測出的點超過四點
            possible_point_list = []
            for i in range(len(approx)):
                possible_point_list.append(approx[i][0])

            a = list(combinations(possible_point_list, 4))
            a = np.asarray(a, np.int).tolist()

            max_area = 0
            for i in range(len(a)):
                area = GetAreaOfPolyGon(a[i])
                if area > max_area:
                    max_area = area
                    index = i

            b = np.asarray(a[index], np.int)
            imgA_tmp3 = cv2.fillPoly(blank_img, [b], (255, 255, 255), cv2.LINE_AA)

        if len(approx) == 4:
            imgA_tmp3 = cv2.fillPoly(blank_img, [approx], (255, 255, 255), cv2.LINE_AA)

        if len(approx) < 4:
            imgA_tmp3 = np.zeros((256, 256), np.uint8)

    imgA_tmp = imgA_tmp3.copy()
    imgAorB_tmp = np.bitwise_or(imgA_tmp, imgB_tmp)
    imgAandB_tmp = np.bitwise_and(imgA_tmp, imgB_tmp)
    _, contoursAorB, hierarchyAorB = cv2.findContours(imgAorB_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, contoursAandB, hierarchyAandB = cv2.findContours(imgAandB_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    area_AorB = 0
    area_AandB = 0

    for cntAorB in range(len(contoursAorB)):
        area_AorB += cv2.contourArea(contoursAorB[cntAorB])

    for cntAandB in range(len(contoursAandB)):
        area_AandB += cv2.contourArea(contoursAandB[cntAandB])

    iou = float(area_AandB / (area_AorB))

    return iou


def target_IOU_cal(predict_img_input_path, GT_img_input_path):

    imgB_list = len(os.listdir(GT_img_input_path))

    iou_list = []

    for i in range(1, imgB_list + 1):
        imgA_path = "{}/test_{:04d}.png".format(predict_img_input_path, i)
        imgB_path = "{}/{}.jpg".format(GT_img_input_path, i)

        imgA = cv2.imread(imgA_path)
        imgB = cv2.imread(imgB_path)

        w = int(imgB.shape[1])
        w2 = int(w / 2)
        imgB_GT = imgB[:, 0:w2]
        iou = iou_cal(imgA, imgB_GT)
        iou_list.append(iou)

    iou_array = np.asarray(iou_list)
    total_iou = iou_array.sum()
    avg_iou = total_iou / len(iou_array)
    return avg_iou
