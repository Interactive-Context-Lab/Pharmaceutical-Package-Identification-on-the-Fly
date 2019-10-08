import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import time
import numpy as np
import os
from model import pix2pix
from RIN_utils.network import PIL
from BCN_args import pix2pix_args
from RIN_args import IFPA_args
from damo_img_preprocess import img_preprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

blister_pack_names_input_path = "./blister_names.txt"
blister_names = []
f = open(blister_pack_names_input_path, 'r')
for l in f:
    l = l.strip('\n')
    blister_names.append(l)
f.close()

# all_imgs_input_path = "D:/baseline_testing_data/test/all"
tf1_imgs_input_path = "D:/baseline_testing_data/test/tf1"
# tf2_imgs_input_path = "D:/baseline_testing_data/test/tf2"

# all_imgs_nemes = os.listdir("{}/left".format(all_imgs_input_path))
tf1_imgs_nemes = os.listdir("{}/left".format(tf1_imgs_input_path))
# tf2_imgs_nemes = os.listdir("{}/left".format(tf2_imgs_input_path))

# a11_left_imgs_list = []
tf1_left_imgs_list = []
# tf2_left_imgs_list = []

# a11_imgs_len = 11500
tf1_imgs_len = 5750
# tf2_imgs_len = 5750

# all_y_true = []
tf1_y_true = []
# tf2_y_true = []

# all_y_pred = []
tf1_y_pred = []
# tf2_y_pred = []

# count = 0
# for i in range(230):
#     for j in range(50):
#         all_y_true.append(count)
#     count += 1

count = 0
for i in range(230):
    for j in range(25):
        tf1_y_true.append(count)
        # tf2_y_true.append(count)
    count += 1

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
    IFPA_args = IFPA_args
    class_num = 230

    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])  # input image size
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    local_x_list, logits_list = PIL(x, training=training_flag, class_num=class_num, drop_rate=IFPA_args.drop_rate,
                                    reduction_ratio=IFPA_args.reduction_ratio,
                                    part=IFPA_args.part, straight=IFPA_args.straight, half=IFPA_args.half,
                                    all=IFPA_args.all).model

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    variables = slim.get_variables_to_restore()

with g2.as_default():
    pix2pix_args = pix2pix_args

with tf.Session(graph=g1, config=tf.ConfigProto(allow_soft_placement=True)) as sess1:
    with tf.Session(graph=g2, config=tf.ConfigProto(allow_soft_placement=True)) as sess2:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        saver.restore(sess1, './checkpoint/IFPA/model.ckpt-58')

        pix2pix_model = pix2pix(sess2, image_size=pix2pix_args.fine_size, batch_size=pix2pix_args.batch_size,
                                output_size=pix2pix_args.fine_size, dataset_name=pix2pix_args.dataset_name,
                                checkpoint_dir=pix2pix_args.checkpoint_dir, sample_dir=pix2pix_args.sample_dir)
        pix2pix_model.demo(pix2pix_args, 0, 0, True)

        input_path = "./test"
        src = np.array([[224, 448], [0, 448], [0, 0], [224, 0]], np.float32)
        dst = np.array([[0, 0], [224, 0], [224, 448], [0, 448]], np.float32)
        M_flip = cv2.getPerspectiveTransform(src, dst)

        ori_frame_size = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], np.float32)
        resize_frame_size = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], np.float32)
        M_resize2ori = cv2.getPerspectiveTransform(resize_frame_size, ori_frame_size)

        img = np.zeros((256, 256, 3), np.uint8)

        tf1_err_output_path = "D:/baseline_error/tf1"
        # tf2_err_output_path = "D:/baseline_error/tf2"

        k = 0
        for step in range(tf1_imgs_len):
            if k % 1000 == 0:
                print("tf1: {}...".format(k))
            k += 1

            frame0 = cv2.imread("{}/left/{}".format(tf1_imgs_input_path, tf1_imgs_nemes[step]))

            frame1 = cv2.imread("{}/right/{}".format(tf1_imgs_input_path, tf1_imgs_nemes[step]))

            img1 = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
            img2 = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)

            result1 = np.concatenate((img, img1), axis=1)
            result2 = np.concatenate((img, img2), axis=1)

            pix2pix_model.demo(pix2pix_args, result1, result2, False)

            img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
            img2_pre = cv2.imread("{}/test_0002.png".format(input_path))

            img1_cal, img1_p, img1_c = img_preprocess(img1_pre, img1, frame0, M_resize2ori)
            img2_cal, img2_p, img2_c = img_preprocess(img2_pre, img2, frame1, M_resize2ori)

            img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))

            result4 = np.concatenate((img1_cal, img2_cal), axis=1)

            RTT_img = cv2.resize(result4, (224, 224), cv2.INTER_AREA)
            RTT_img = cv2.cvtColor(RTT_img, cv2.COLOR_BGR2RGB)
            RTT_img = RTT_img.reshape(1, 224, 224, 3)

            pred = sess1.run([logits_list], feed_dict={
                x: RTT_img,
                training_flag: False
            })

            RTT_pred = np.argmax(pred[0][0], 1)
            tf1_y_pred.append(RTT_pred[0])

            if tf1_y_true[step] != tf1_y_pred[step]:
                cv2.imwrite(
                    "{}/{:06d}_true_{:03d}_pred_{:03d}.jpg".format(tf1_err_output_path, step, tf1_y_true[step],
                                                                   tf1_y_pred[step]),
                    result4)

        accuracy = accuracy_score(tf1_y_true, tf1_y_pred)
        precision = precision_score(tf1_y_true, tf1_y_pred, average='macro')
        recall = recall_score(tf1_y_true, tf1_y_pred, average='macro')
        F1_score = f1_score(tf1_y_true, tf1_y_pred, average='macro')

        record1 = "tf1: accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}\n".format(accuracy, precision,
                                                                                                    recall, F1_score)
        print(record1)

        # k = 0
        # for step in range(tf2_imgs_len):
        #     if k % 1000 == 0:
        #         print("tf2: {}...".format(k))
        #     k += 1
        #
        #     frame0 = cv2.imread("{}/left/{}".format(tf2_imgs_input_path, tf2_imgs_nemes[step]))
        #
        #     frame1 = cv2.imread("{}/right/{}".format(tf2_imgs_input_path, tf2_imgs_nemes[step]))
        #
        #     img1 = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
        #     img2 = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)
        #
        #     result1 = np.concatenate((img, img1), axis=1)
        #     result2 = np.concatenate((img, img2), axis=1)
        #
        #     pix2pix_model.demo(pix2pix_args, result1, result2, False)
        #
        #     img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
        #     img2_pre = cv2.imread("{}/test_0002.png".format(input_path))
        #
        #     img1_cal, img1_p, img1_c = img_preprocess(img1_pre, img1, frame0, M_resize2ori)
        #     img2_cal, img2_p, img2_c = img_preprocess(img2_pre, img2, frame1, M_resize2ori)
        #
        #     img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))
        #
        #     result4 = np.concatenate((img1_cal, img2_cal), axis=1)
        #
        #     RTT_img = cv2.resize(result4, (224, 224), cv2.INTER_AREA)
        #     RTT_img = cv2.cvtColor(RTT_img, cv2.COLOR_BGR2RGB)
        #     RTT_img = RTT_img.reshape(1, 224, 224, 3)
        #
        #     pred = sess1.run([logits_list], feed_dict={
        #         x: RTT_img,
        #         training_flag: False
        #     })
        #
        #     RTT_pred = np.argmax(pred[0][0], 1)
        #     tf2_y_pred.append(RTT_pred[0])
        #
        #     if tf2_y_true[step] != tf2_y_pred[step]:
        #         cv2.imwrite(
        #             "{}/{:06d}_true_{:03d}_pred_{:03d}.jpg".format(tf2_err_output_path, step, tf2_y_true[step],
        #                                                            tf2_y_pred[step]),
        #             result4)
        #
        # accuracy = accuracy_score(tf2_y_true, tf2_y_pred)
        # precision = precision_score(tf2_y_true, tf2_y_pred, average='macro')
        # recall = recall_score(tf2_y_true, tf2_y_pred, average='macro')
        # F1_score = f1_score(tf2_y_true, tf2_y_pred, average='macro')
        #
        # record2 = "tf2: accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}\n".format(accuracy, precision,
        #                                                                                             recall, F1_score)
        # print(record2)

        # k = 0
        # for step in range(a11_imgs_len):
        #     if k % 1000 == 0:
        #         print("all: {}...".format(k))
        #     k += 1
        #
        #     frame0 = cv2.imread("{}/left/{}".format(all_imgs_input_path, all_imgs_nemes[step]))
        #
        #     frame1 = cv2.imread("{}/right/{}".format(all_imgs_input_path, all_imgs_nemes[step]))
        #
        #     img1 = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
        #     img2 = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)
        #
        #     result1 = np.concatenate((img, img1), axis=1)
        #     result2 = np.concatenate((img, img2), axis=1)
        #
        #     pix2pix_model.demo(pix2pix_args, result1, result2, False)
        #
        #     img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
        #     img2_pre = cv2.imread("{}/test_0002.png".format(input_path))
        #
        #     img1_cal, img1_p, img1_c = img_preprocess(img1_pre, img1, frame0, M_resize2ori)
        #     img2_cal, img2_p, img2_c = img_preprocess(img2_pre, img2, frame1, M_resize2ori)
        #
        #     img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))
        #
        #     result4 = np.concatenate((img1_cal, img2_cal), axis=1)
        #
        #     RTT_img = cv2.resize(result4, (224, 224), cv2.INTER_AREA)
        #     RTT_img = cv2.cvtColor(RTT_img, cv2.COLOR_BGR2RGB)
        #     RTT_img = RTT_img.reshape(1, 224, 224, 3)
        #
        #     pred = sess1.run([logits_list], feed_dict={
        #         x: RTT_img,
        #         training_flag: False
        #     })
        #
        #     RTT_pred = np.argmax(pred[0][0], 1)
        #     all_y_pred.append(RTT_pred[0])
        #
        # accuracy = accuracy_score(all_y_true, all_y_pred)
        # precision = precision_score(all_y_true, all_y_pred, average='macro')
        # recall = recall_score(all_y_true, all_y_pred, average='macro')
        # F1_score = f1_score(all_y_true, all_y_pred, average='macro')
        #
        # record = "all: accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}\n".format(accuracy, precision,
        #                                                                                            recall, F1_score)
        # print(record1)
        # print(record2)
        # print(record)

        coord.request_stop()
        coord.join(threads)
