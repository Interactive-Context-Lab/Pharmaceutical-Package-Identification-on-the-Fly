import argparse
import os
import scipy.misc
import numpy as np

from model import pix2pix
import tensorflow as tf
import cv2
import math
from itertools import combinations
from cGAN_BGandTARGET_IOU_cal import target_IOU_cal
import time
import shutil

from load_testing_data import load_imgs



parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_name', dest='dataset_name', default='facades', help='name of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')

args = parser.parse_args()

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


def iou_cal(imgA, imgA_color, ori_frame, M_resize2ori):
    imgA_tmp = imgA.copy()

    imgA_tmp = cv2.cvtColor(imgA_tmp, cv2.COLOR_BGR2GRAY)

    retA, imgA_tmp = cv2.threshold(imgA_tmp, 127, 255, cv2.THRESH_BINARY)

    _,contoursA, hierarchyA = cv2.findContours(imgA_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0  # 找出最大面積
    index = 0
    for cntA in range(len(contoursA)):
        area = cv2.contourArea(contoursA[cntA])
        if area > max_area:
            max_area = area
            index = cntA

    blank_img = np.zeros(imgA_tmp.shape, np.uint8)
    blank_img1 = np.zeros(imgA_tmp.shape, np.uint8)
    hull2 = cv2.convexHull(contoursA[index])
    cv2.drawContours(blank_img1, [hull2], 0, (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

    a = cv2.cvtColor(blank_img1, cv2.COLOR_GRAY2BGR)

    epsilon = 0.01 * cv2.arcLength(hull2, True)
    approx = cv2.approxPolyDP(hull2, epsilon, True)
    cv2.drawContours(a, approx, -1, (0, 0, 255), 3)

    if len(approx) > 4:  # 偵測出的點超過四點
        point_pairing = []  # 用來儲存配對點
        point_pairing_distance = []  # 用來記錄配對點之間的距離
        for i in range(len(approx)):  # 紀錄所有對角的配對點和距離
            left_p = i - 1
            if left_p < 0:
                left_p = len(approx) - 1
            right_p = i + 1
            if right_p > len(approx) - 1:
                right_p = 0
            for j in range(len(approx)):
                if j != left_p and j > i and j != right_p:  # j > i: 排除一樣的配對點
                    diff = approx[i] - approx[j]
                    lengh = math.hypot(diff[0][0], diff[0][1])
                    point_pairing.append([i, j, lengh])
                    point_pairing_distance.append(lengh)

        point_pairing1 = point_pairing_distance.index(max(point_pairing_distance))  # 紀錄具有最大距離的配對點index

        another_two_points = []  # 用來記錄其餘所有點的可能配對
        for i in range(len(approx)):  # 找出其餘所有點的可能配對
            if i != point_pairing[point_pairing1][0] and i != point_pairing[point_pairing1][1]:
                for j in range(len(approx)):
                    if j != point_pairing[point_pairing1][0] and j != point_pairing[point_pairing1][1] and j > i:
                        another_two_points.append([i, j])
        max_area = 0
        for i in range(len(another_two_points)):  # 將與配對點有相連接的配對點刪除，並找到四點組合面積最大的點配對
            points_array = np.array([[point_pairing[point_pairing1][0],
                                      point_pairing[point_pairing1][1],
                                      another_two_points[i][0],
                                      another_two_points[i][1]]], np.int)
            points_array_arg = np.argsort(points_array)

            p0_index = points_array[0][points_array_arg[0][0]]
            p1_index = points_array[0][points_array_arg[0][1]]
            p2_index = points_array[0][points_array_arg[0][2]]
            p3_index = points_array[0][points_array_arg[0][3]]

            points_list = [approx[p0_index][0],
                           approx[p1_index][0],
                           approx[p2_index][0],
                           approx[p3_index][0]]
            area = GetAreaOfPolyGon(points_list)
            if area > max_area:
                max_area = area
                final_four_points_index = [p0_index, p1_index, p2_index, p3_index]

        four_points = np.array(
            [approx[final_four_points_index[0]][0],
             approx[final_four_points_index[1]][0],
             approx[final_four_points_index[2]][0],
             approx[final_four_points_index[3]][0]], np.float32)

        point_int = np.array(
            [approx[final_four_points_index[0]][0],
             approx[final_four_points_index[1]][0],
             approx[final_four_points_index[2]][0],
             approx[final_four_points_index[3]][0]], np.int)

        a = np.zeros((256,256,3), np.uint8)
        b = imgA_color.copy()
        imgA_tmp2 = cv2.polylines(b, [point_int], True, (0,0,255), 2, cv2.LINE_AA)
        imgA_tmp3 = cv2.fillPoly(a, [point_int], (255,255,255), cv2.LINE_AA)

        p0_p1 = (four_points[0] - four_points[1])
        p1_p2 = (four_points[1] - four_points[2])
        p2_p3 = (four_points[2] - four_points[3])
        p3_p0 = (four_points[3] - four_points[0])


        lengh1 = math.hypot(p0_p1[0], p0_p1[1]) + math.hypot(p2_p3[0], p2_p3[1])

        lengh2 = math.hypot(p1_p2[0], p1_p2[1]) + math.hypot(p3_p0[0], p3_p0[1])

        if lengh1 > lengh2:
            four_points = np.array(
                [approx[final_four_points_index[3]][0],
                 approx[final_four_points_index[0]][0],
                 approx[final_four_points_index[1]][0],
                 approx[final_four_points_index[2]][0]], np.float32)

        z = np.ones((4, 1), np.float32)
        four_points_tmp = np.concatenate((four_points, z), axis=1)
        four_points_tmp_trans = four_points_tmp.transpose()
        four_points = np.matmul(M_resize2ori, four_points_tmp_trans)
        four_points_op_t = four_points.transpose()
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

    elif len(approx) == 4:
        four_points = np.array(
            [approx[0][0],
             approx[1][0],
             approx[2][0],
             approx[3][0]], np.float32)

        point_int = np.array(
            [approx[0][0],
             approx[1][0],
             approx[2][0],
             approx[3][0]], np.int)

        a = np.zeros((256, 256, 3), np.uint8)
        b = imgA_color.copy()
        imgA_tmp2 = cv2.polylines(b, [point_int], True, (0, 0, 255), 2, cv2.LINE_AA)
        imgA_tmp3 = cv2.fillPoly(a, [point_int], (255, 255, 255), cv2.LINE_AA)

        p0_p1 = (four_points[0] - four_points[1])
        p1_p2 = (four_points[1] - four_points[2])
        p2_p3 = (four_points[2] - four_points[3])
        p3_p0 = (four_points[3] - four_points[0])

        lengh1 = math.hypot(p0_p1[0], p0_p1[1]) + math.hypot(p2_p3[0], p2_p3[1])

        lengh2 = math.hypot(p1_p2[0], p1_p2[1]) + math.hypot(p3_p0[0], p3_p0[1])

        if lengh1 > lengh2:
            four_points = np.array(
                [approx[3][0],
                 approx[0][0],
                 approx[1][0],
                 approx[2][0]], np.float32)

        z = np.ones((4, 1), np.float32)
        four_points_tmp = np.concatenate((four_points, z), axis=1)
        four_points_tmp_trans = four_points_tmp.transpose()
        four_points = np.matmul(M_resize2ori, four_points_tmp_trans)
        four_points_op_t = four_points.transpose()
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


    elif len(approx) < 4:
        warp = np.zeros((224, 448, 3), np.uint8)
        imgA_tmp3 = np.zeros((256, 256, 3), np.uint8)

    return warp, imgA_tmp3, imgA_tmp2


def img_preprocess(imgA, imgA_color, ori_frame, M_resize2ori):
    imgA_tmp = imgA.copy()

    imgA_tmp = cv2.cvtColor(imgA_tmp, cv2.COLOR_BGR2GRAY)

    retA, imgA_tmp = cv2.threshold(imgA_tmp, 127, 255, cv2.THRESH_BINARY)

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


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = pix2pix(sess, image_size=args.fine_size, batch_size=args.batch_size,
                        output_size=args.fine_size, dataset_name=args.dataset_name,
                        checkpoint_dir=args.checkpoint_dir, sample_dir=args.sample_dir)

        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        elif args.phase == 'demo':
            model.demo(args, 0, 0, True)
            cap0 = cv2.VideoCapture(1)
            cap1 = cv2.VideoCapture(0)
            cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            input_path = "./test"
            src = np.array([[224, 448], [0, 448], [0, 0], [224, 0]], np.float32)
            dst = np.array([[0, 0], [224, 0], [224, 448], [0, 448]], np.float32)
            M_flip = cv2.getPerspectiveTransform(src, dst)

            ori_frame_size = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], np.float32)
            resize_frame_size = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], np.float32)
            M_resize2ori = cv2.getPerspectiveTransform(resize_frame_size, ori_frame_size)
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # videoWriter = cv2.VideoWriter('./video/test.mp4', fourcc, 30.0, (448, 448))

            img = np.zeros((256, 256, 3), np.uint8)

            #**************bg_sub_color*********************************************************
            # count = 1
            # global frame0_bg, frame1_bg
            # while (True):
            #     ret0, frame0 = cap0.read()
            #     assert ret0
            #     ret1, frame1 = cap1.read()
            #     assert ret1
            #     if (count > 10):
            #         frame0_bg = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
            #         frame1_bg = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)
            #         break
            #     count += 1
            # **************bg_sub_color*********************************************************

            while (True):
                st = time.time()
                ret0, frame0 = cap0.read()
                assert ret0
                ret1, frame1 = cap1.read()
                assert ret1

                img1 = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
                img2 = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)

                # **************bg_sub_color*********************************************************
                # img1 = cv2.subtract(img1, frame0_bg)
                # img2 = cv2.subtract(img2, frame1_bg)
                # **************bg_sub_color*********************************************************


                # result = np.concatenate((img1, img2), axis=1)
                result1 = np.concatenate((img, img1), axis=1)
                result2 = np.concatenate((img, img2), axis=1)
                st1 = time.time()
                model.demo(args, result1, result2, False)
                print("model cost time:{:.2f}s".format(time.time() - st1))
                print("cost time:{:.2f}s".format(time.time() - st))
                img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
                img2_pre = cv2.imread("{}/test_0002.png".format(input_path))

                img1_cal, img1_p, img1_c = img_preprocess(img1_pre, img1, frame0, M_resize2ori)
                img2_cal, img2_p, img2_c = img_preprocess(img2_pre, img2, frame1, M_resize2ori)
                img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))


                result3 = np.concatenate((img1_pre, img2_pre), axis=1)
                result4 = np.concatenate((img1_cal, img2_cal), axis=1)
                # videoWriter.write(result4)
                result5 = np.concatenate((img1_p, img2_p), axis=1)
                result6 = np.concatenate((img1_c, img2_c), axis=1)

                result7 = np.concatenate((result6, result3))
                result7 = np.concatenate((result7, result5))
                blank = np.zeros((448, 320, 3), np.uint8)
                result4 = np.concatenate((result4, blank), axis=1)
                blank = np.zeros((320, 768, 3), np.uint8)
                result4 = np.concatenate((result4, blank), axis=0)
                result8 = np.concatenate((result7, result4), axis=1)
                # cv2.imshow("0", result)
                cv2.imshow("1", result8)
                # cv2.imshow("2", result4)

                key = cv2.waitKey(30) & 0xFF
                if (key == 27):
                    cap0.release()
                    cap1.release()
                    # videoWriter.release()
                    cv2.destroyAllWindows()
                    break
        elif args.phase == 'test_IOU':
            # iou_imfor = []
            with open("./{}_test_record.txt".format(args.dataset_name), 'a') as fp:
                fp.write("model\t, avg_IOU\n")

            diffBG_path = './datasets/{}/{}/*.jpg'.format(args.dataset_name, "val")

            is_grayscale = (args.input_nc == 1)
            diffBG_images = load_imgs(diffBG_path, is_grayscale, args.batch_size)

            for i in range(1, 201):
                model.test_IOU(args, sample_images=diffBG_images, model_name='pix2pix.model-{}'.format(i), stat=True)
                avg_iou = target_IOU_cal('{}'.format(args.test_dir), './datasets/{}/{}'.format(args.dataset_name, "val"))

                print("test1: model-{}, avg_IOU:{:.2f}".format(i, avg_iou))


                test_record = "{}\t, {:.2f}".format(i, avg_iou)

                with open("./{}_test_record.txt".format(args.dataset_name), 'a') as fp:
                    fp.write("{}\n".format(test_record))




if __name__ == '__main__':
    tf.app.run()
