import cv2
import math
import numpy as np
from itertools import combinations


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


def img_preprocess(imgA, imgA_color, ori_frame, M_resize2ori):
    imgA_tmp = imgA.copy()

    imgA_tmp = cv2.cvtColor(imgA_tmp, cv2.COLOR_BGR2GRAY)

    retA, imgA_tmp = cv2.threshold(imgA_tmp, 240, 255, cv2.THRESH_BINARY)

    _, contoursA, hierarchyA = cv2.findContours(imgA_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0  # 找出最大面積
    index = 0
    for cntA in range(len(contoursA)):
        area = cv2.contourArea(contoursA[cntA])
        if area > max_area:
            max_area = area
            index = cntA

    if max_area < 50 or len(contoursA[index]) < 3 or len(contoursA) < 1:
        warp = np.zeros((448, 224, 3), np.uint8)
        imgA_tmp3 = np.zeros((256, 256, 3), np.uint8)
        imgA_tmp2 = imgA_color.copy()
    else:
        blank_img = np.zeros(imgA_tmp.shape, np.uint8)
        hull2 = cv2.convexHull(contoursA[index])
        cv2.drawContours(blank_img, [hull2], 0, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

        a = cv2.cvtColor(blank_img, cv2.COLOR_GRAY2BGR)

        epsilon = 0.01 * cv2.arcLength(hull2, True)
        approx = cv2.approxPolyDP(hull2, epsilon, True)
        cv2.drawContours(a, approx, -1, (0, 0, 255), 3)

        if len(approx) < 4:
            warp = np.zeros((448, 224, 3), np.uint8)
            imgA_tmp3 = np.zeros((256, 256, 3), np.uint8)
            imgA_tmp2 = imgA_color.copy()
        else:
            possible_point_list = []
            for i in range(len(approx)):
                possible_point_list.append(approx[i][0])

            possible_point_combinations_list = list(combinations(possible_point_list, 4))
            possible_point_combinations_list = np.asarray(possible_point_combinations_list, np.int).tolist()

            max_area = 0
            for i in range(len(possible_point_combinations_list)):
                area = GetAreaOfPolyGon(possible_point_combinations_list[i])
                if area > max_area:
                    max_area = area
                    index = i

            max_area_point_combinations = np.asarray(possible_point_combinations_list[index], np.int)
            a = np.zeros(imgA.shape, np.uint8)
            imgA_tmp3 = cv2.fillPoly(a, [max_area_point_combinations], (255, 255, 255), cv2.LINE_AA)
            b = imgA_color.copy()
            imgA_tmp2 = cv2.polylines(b, [max_area_point_combinations], True, (0, 0, 255), 2,
                                      cv2.LINE_AA)

            max_area_point_combinations_float = max_area_point_combinations.astype(np.float32)
            p0_p1 = (max_area_point_combinations_float[0] - max_area_point_combinations_float[1])
            p1_p2 = (max_area_point_combinations_float[1] - max_area_point_combinations_float[2])
            p2_p3 = (max_area_point_combinations_float[2] - max_area_point_combinations_float[3])
            p3_p0 = (max_area_point_combinations_float[3] - max_area_point_combinations_float[0])

            lengh1 = math.hypot(p0_p1[0], p0_p1[1]) + math.hypot(p2_p3[0], p2_p3[1])

            lengh2 = math.hypot(p1_p2[0], p1_p2[1]) + math.hypot(p3_p0[0], p3_p0[1])
            if lengh1 > lengh2:
                max_area_point_combinations_float = np.array(
                    [max_area_point_combinations[3],
                     max_area_point_combinations[0],
                     max_area_point_combinations[1],
                     max_area_point_combinations[2]], np.float32)

            z = np.ones((4, 1), np.float32)
            max_area_point_combinations_float_tmp = np.concatenate((max_area_point_combinations_float, z), axis=1)
            max_area_point_combinations_float_tmp_trans = max_area_point_combinations_float_tmp.transpose()
            max_area_point_combinations_float_ori = np.matmul(M_resize2ori, max_area_point_combinations_float_tmp_trans)
            four_points_op_t = max_area_point_combinations_float_ori.transpose()
            four_points = np.array([[round((four_points_op_t[0][0] / four_points_op_t[0][2]), 1),
                                     round((four_points_op_t[0][1] / four_points_op_t[0][2]), 1)],
                                    [round((four_points_op_t[1][0] / four_points_op_t[1][2]), 1),
                                     round((four_points_op_t[1][1] / four_points_op_t[1][2]), 1)],
                                    [round((four_points_op_t[2][0] / four_points_op_t[2][2]), 1),
                                     round((four_points_op_t[2][1] / four_points_op_t[2][2]), 1)],
                                    [round((four_points_op_t[3][0] / four_points_op_t[3][2]), 1),
                                     round((four_points_op_t[3][1] / four_points_op_t[3][2]), 1)]], np.float32)

            dst = np.array([[0, 0], [224, 0], [224, 448], [0, 448]], np.float32)
            M = cv2.getPerspectiveTransform(four_points, dst)
            warp = cv2.warpPerspective(ori_frame, M, (224, 448))

    return warp, imgA_tmp3, imgA_tmp2