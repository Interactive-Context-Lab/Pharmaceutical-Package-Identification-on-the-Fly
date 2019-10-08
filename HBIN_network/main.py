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

        cap0 = cv2.VideoCapture(0)
        cap1 = cv2.VideoCapture(2)
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

        img = np.zeros((256, 256, 3), np.uint8)
        while (True):
            st = time.time()
            ret0, frame0 = cap0.read()
            assert ret0
            ret1, frame1 = cap1.read()
            assert ret1

            img1 = cv2.resize(frame0, (256, 256), cv2.INTER_AREA)
            img2 = cv2.resize(frame1, (256, 256), cv2.INTER_AREA)

            result1 = np.concatenate((img, img1), axis=1)
            result2 = np.concatenate((img, img2), axis=1)
            print("load img time = {:.4f}s".format(time.time() - st))

            pix2pix_model.demo(pix2pix_args, result1, result2, False)


            img1_pre = cv2.imread("{}/test_0001.png".format(input_path))
            img2_pre = cv2.imread("{}/test_0002.png".format(input_path))


            img1_cal, img1_p, img1_c = img_preprocess(img1_pre, img1, frame0, M_resize2ori)
            img2_cal, img2_p, img2_c = img_preprocess(img2_pre, img2, frame1, M_resize2ori)

            img2_cal = cv2.warpPerspective(img2_cal, M_flip, (224, 448))

            result3 = np.concatenate((img1_pre, img2_pre), axis=1)
            result4 = np.concatenate((img1_cal, img2_cal), axis=1)

            RTT_img = cv2.resize(result4, (224, 224), cv2.INTER_AREA)
            RTT_img = cv2.cvtColor(RTT_img, cv2.COLOR_BGR2RGB)
            RTT_img = RTT_img.reshape(1, 224, 224, 3)

            st3 = time.time()
            pred = sess1.run([logits_list], feed_dict={
                x: RTT_img,
                training_flag: False
            })

            RTT_pred = np.argmax(pred[0][0], 1)
            RTT_pred_name = blister_names[RTT_pred[0]]

            result5 = np.concatenate((img1_p, img2_p), axis=1)
            result6 = np.concatenate((img1_c, img2_c), axis=1)

            result7 = np.concatenate((result6, result3))
            result7 = np.concatenate((result7, result5))
            blank = np.zeros((448, 320, 3), np.uint8)
            result4 = np.concatenate((result4, blank), axis=1)
            blank = np.zeros((320, 768, 3), np.uint8)
            result4 = np.concatenate((result4, blank), axis=0)
            result8 = np.concatenate((result7, result4), axis=1)

            cv2.putText(result8, RTT_pred_name, (500, 600), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 1)
            cv2.imshow("1", result8)

            key = cv2.waitKey(30) & 0xFF
            if (key == 27):
                cap0.release()
                cap1.release()
                cv2.destroyAllWindows()
                break

        coord.request_stop()
        coord.join(threads)
