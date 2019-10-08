import tensorflow.contrib.slim as slim
import tensorflow as tf
import cv2
import time
import numpy as np
from model import pix2pix
from RIN_utils.network import PIL
from BCN_args import pix2pix_args
from RIN_args import IFPA_args
from damo_img_preprocess import img_preprocess
from sklearn.metrics import precision_score, recall_score, f1_score
import os


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


blister_pack_names_input_path = "./blister_names.txt"
blister_names = []
f = open(blister_pack_names_input_path, 'r')
for l in f:
    l = l.strip('\n')
    blister_names.append(l)
f.close()

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

        saver.restore(sess1, './checkpoint/RIN/model.ckpt-98')

        pix2pix_model = pix2pix(sess2, image_size=pix2pix_args.fine_size, batch_size=pix2pix_args.batch_size,
                                output_size=pix2pix_args.fine_size, dataset_name=pix2pix_args.dataset_name,
                                checkpoint_dir=pix2pix_args.checkpoint_dir, sample_dir=pix2pix_args.sample_dir)
        pix2pix_model.demo(pix2pix_args, 0, 0, True)

        src_path = "./HBIN_bg_sub_testdata/blister_img/"
        bg_src_path = "./HBIN_bg_sub_testdata/blister_sub_img/"

        # label_path = "/home/ee303/Documents/baseline_demo/blister_tf3/label_img"
        input_path = "./test"
        output_path = "./HBIN_error/part-level"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        src = np.array([[224, 448], [0, 448], [0, 0], [224, 0]], np.float32)
        dst = np.array([[0, 0], [224, 0], [224, 448], [0, 448]], np.float32)
        M_flip = cv2.getPerspectiveTransform(src, dst)

        ori_frame_size = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], np.float32)
        resize_frame_size = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], np.float32)
        M_resize2ori = cv2.getPerspectiveTransform(resize_frame_size, ori_frame_size)
        count = 1
        img = np.zeros((256, 256, 3), np.uint8)
        y_pred = []
        y_true = []
        counter = 0

        for name in blister_names:
            print(name)
            for id in range(1, 26):

                # start_time = time.time()
                # frame0 = cv2.imread("{}/{}/R/img_{}.jpg".format(bg_src_path, name, id))
                # frame1 = cv2.imread("{}/{}/L/img_{}.jpg".format(bg_src_path, name, id))
                # frame0 = cv2.imread("{}/{}/{:06d}_R.jpg".format(src_path, name, id))
                # frame1 = cv2.imread("{}/{}/{:06d}_L.jpg".format(src_path, name, id))

                frame0_color = cv2.imread("{}/{}/R/img_{}.jpg".format(src_path, name, id))
                frame1_color = cv2.imread("{}/{}/L/img_{}.jpg".format(src_path, name, id))
                # img1_color = cv2.resize(frame0_color, (256, 256), cv2.INTER_AREA)
                # img2_color = cv2.resize(frame1_color, (256, 256), cv2.INTER_AREA)

                img1 = cv2.resize(frame0_color, (256, 256), cv2.INTER_AREA)
                img2 = cv2.resize(frame1_color, (256, 256), cv2.INTER_AREA)

                result1 = np.concatenate((img, img1), axis=1)
                result2 = np.concatenate((img, img2), axis=1)

                pix2pix_model.demo(pix2pix_args, result1, result2, False)
                # end_time = time.time()
                # print("consuming time:" + str(end_time - start_time))
                # img1_pre = cv2.imread("{}/{}/{:06d}_L.jpg".format(label_path, name, id))
                # img1_pre = cv2.resize(img1_pre, (256, 256), cv2.INTER_AREA)
                # img2_pre = cv2.imread("{}/{}/{:06d}_R.jpg".format(label_path, name, id))
                # img2_pre = cv2.resize(img2_pre, (256, 256), cv2.INTER_AREA)
                img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
                img2_pre = cv2.imread("{}/test_0002.png".format(input_path))

                st2 = time.time()
                img1_cal, img1_p, img1_c = img_preprocess(img1_pre, img1, frame0_color, M_resize2ori)
                img2_cal, img2_p, img2_c = img_preprocess(img2_pre, img2, frame1_color, M_resize2ori)

                st4 = time.time()
                img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))

                result3 = np.concatenate((img1_pre, img2_pre), axis=1)
                result4 = np.concatenate((img1_cal, img2_cal), axis=1)
                # cv2.imshow("1",result4)
                # print(result4.shape)
                # cv2.waitKey(0)
                # cv2.imwrite("./tf3RTT/"+str(counter)+".jpg",result4)
                counter+=1

                # RTT recognition
                RTT_img = cv2.resize(result4, (224, 224), cv2.INTER_AREA)
                RTT_img = cv2.cvtColor(RTT_img, cv2.COLOR_BGR2RGB)
                RTT_img = RTT_img.reshape(1, 224, 224, 3)

                pred = sess1.run([logits_list], feed_dict={
                    x: RTT_img,
                    training_flag: False
                })

                np_pred = np.asarray(pred)
                mean_pred = np.mean(np_pred[0], axis=0)

                softmax_x = np.asarray(mean_pred).reshape(-1).tolist()
                softmax_x = softmax(softmax_x)
                softmax_x = softmax_x.reshape(-1, 1)

                RTT_pred = np.argmax(mean_pred[0], 0)
                RTT_pred_confidence = softmax_x[RTT_pred][0]
                pred_name = blister_names[RTT_pred]
                RTT_pred_name = "{}% {}".format(int(RTT_pred_confidence * 100), blister_names[RTT_pred])


                result5 = np.concatenate((img1_p, img2_p), axis=1)
                result6 = np.concatenate((img1_c, img2_c), axis=1)

                result7 = np.concatenate((result6, result3))
                result7 = np.concatenate((result7, result5))
                blank = np.zeros((448, 320, 3), np.uint8)
                result4 = np.concatenate((result4, blank), axis=1)
                blank = np.zeros((320, 768, 3), np.uint8)
                result4 = np.concatenate((result4, blank), axis=0)
                result8 = np.concatenate((result7, result4), axis=1)

                cv2.putText(result8, RTT_pred_name, (500, 600), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
                y_true.append(blister_names.index(name))
                y_pred.append(blister_names.index(pred_name))


                if blister_names.index(name) != blister_names.index(pred_name):
                    cv2.imwrite("{}/true_{:03d}_pred_{:03d}_{:06d}.jpg".format(output_path, blister_names.index(name),
                                                                        blister_names.index(pred_name), count), result8)
                count += 1
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1_score = f1_score(y_true, y_pred, average='macro')
        print(precision)
        print(recall)
        print(F1_score)

        coord.request_stop()
        coord.join(threads)
