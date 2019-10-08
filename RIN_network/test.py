import tensorflow.contrib.slim as slim
import argparse
import time
import numpy as np

from utils.read_decode import *
from utils.network import PIL
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

######################################################################
# Options
# --------
parser = argparse.ArgumentParser()
parser.add_argument('--record_dir', type=str, default='./datasets/blister_pack_RTT/datas/tfrecords/')
parser.add_argument('--feature_dir', type=str, default='./datasets/blister_pack_RTT/features/')
parser.add_argument('--model_dir', type=str, default='./checkpoint/model/')
parser.add_argument('--pre_model', type=str, default='')
parser.add_argument('--restore_model', type=str, default='./checkpoint/model/190704_115244/model.ckpt-97')
parser.add_argument('--dataset', type=str, default='blister_pack_RTT')
parser.add_argument('--batch_size', type=int, default=1 )
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--reduction_ratio', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001)  # 0.001
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--drop_rate', type=float, default=0.2)
parser.add_argument('--part', type=bool, default=True)
parser.add_argument('--all', type=bool, default=True)
parser.add_argument('--half', type=bool, default=False)
parser.add_argument('--straight', type=bool, default=False)
parser.add_argument('--only_test', type=bool, default=True)
parser.add_argument('--load_model', type=bool, default=True)
parser.add_argument('--feat_ext', type=bool, default=True)
parser.add_argument('--normalize_feat', type=bool, default=True)
parser.add_argument('--rerank', type=bool, default=True)
args = parser.parse_args()

all_test_file = args.record_dir + 'test_all.tfrecords'

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

class_num = 230
f1_test_samples = 4140
f2_test_samples = 4140
all_test_samples = 8180

reduction_ratio = args.reduction_ratio
batch_size = args.batch_size
drop_rate = args.drop_rate


all_test_x, all_test_y = read_and_decode_test(all_test_file, batch_size)
all_test_y = tf.one_hot(all_test_y, class_num)


x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])  # input image size
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

local_x_list, logits_list = PIL(x, training=training_flag, class_num=class_num, drop_rate=args.drop_rate,
                                reduction_ratio=args.reduction_ratio,
                                part=args.part, straight=args.straight, half=args.half, all=args.all).model

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

variables = slim.get_variables_to_restore()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print('Initialing all variables......')
    # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('Finished initialing.')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # print('Start Testing...{}'.format(f1_test_file))
    # iteration = f1_test_samples // batch_size
    # test_st = time.time()
    # total_iter = 0
    # f1_test_metrics = []
    # for epoch in range(1, args.total_epochs + 1):
    #     epoch_st = time.time()
    #     print('Loading model......')
    #     st = time.time()
    #     print('Restore from model.ckpt-{}'.format(epoch))
    #     saver.restore(sess, './checkpoint/model/190429_192350/model.ckpt-{}'.format(epoch))
    #     print('Finished loading, {:.2f}s\n'.format(time.time() - st))
    #
    #     step_st = time.time()
    #     all_y_pred = []
    #     all_y_true = []
    #     for step in range(1, iteration + 1):
    #         batch_x, batch_y = sess.run([f1_test_x, f1_test_y])
    #         test_feed_dict = {
    #             x: batch_x,
    #             label: batch_y,
    #             training_flag: False
    #         }
    #         pred = sess.run([logits_list], feed_dict=test_feed_dict)
    #         pred = np.asarray(pred)
    #         y_pred = np.argmax(pred[0][0], 1)
    #         y_true = np.argmax(batch_y, 1)
    #         all_y_pred.extend(y_pred)
    #         all_y_true.extend(y_true)
    #         total_iter += 1
    #         if step % 100 == 0:
    #             print('Ep {}, Iter {}, Step {}/{}, {:.2f}s'.format(epoch, total_iter, step, iteration,
    #                                                                time.time() - step_st))
    #             step_st = time.time()
    #
    #     accuracy = accuracy_score(all_y_true, all_y_pred)
    #     precision = precision_score(all_y_true, all_y_pred, average='macro')
    #     recall = recall_score(all_y_true, all_y_pred, average='macro')
    #     F1_score = f1_score(all_y_true, all_y_pred, average='macro')
    #
    #     print('epoch: {}/{}, Epoch testing time {:.2f}s, Total testing time {:.2f}s'.format(epoch, args.total_epochs,
    #                                                                                         time.time() - epoch_st,
    #                                                                                         time.time() - test_st))
    #     f1_test_metrics.append([accuracy, precision, recall, F1_score])
    #     record = "model-{:03d}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}\n".format(epoch,
    #                                                                                                         accuracy,
    #                                                                                                         precision,
    #                                                                                                         recall,
    #                                                                                                         F1_score)
    #     print(record)
    #     with open("./f1_classification_metric_record.txt", 'a') as fp:
    #         fp.write(record)
    #
    # print('Start Testing...{}'.format(f2_test_file))
    # iteration = f2_test_samples // batch_size
    # test_st = time.time()
    # total_iter = 0
    # f2_test_metrics = []
    # for epoch in range(1, args.total_epochs + 1):
    #     epoch_st = time.time()
    #     print('Loading model......')
    #     st = time.time()
    #     print('Restore from model.ckpt-{}'.format(epoch))
    #     saver.restore(sess, './checkpoint/model/190429_192350/model.ckpt-{}'.format(epoch))
    #     print('Finished loading, {:.2f}s\n'.format(time.time() - st))
    #
    #     step_st = time.time()
    #     all_y_pred = []
    #     all_y_true = []
    #     for step in range(1, iteration + 1):
    #         batch_x, batch_y = sess.run([f2_test_x, f2_test_y])
    #         test_feed_dict = {
    #             x: batch_x,
    #             label: batch_y,
    #             training_flag: False
    #         }
    #         pred = sess.run([logits_list], feed_dict=test_feed_dict)
    #         pred = np.asarray(pred)
    #         y_pred = np.argmax(pred[0][0], 1)
    #         y_true = np.argmax(batch_y, 1)
    #         all_y_pred.extend(y_pred)
    #         all_y_true.extend(y_true)
    #         total_iter += 1
    #         if step % 100 == 0:
    #             print('Ep {}, Iter {}, Step {}/{}, {:.2f}s'.format(epoch, total_iter, step, iteration,
    #                                                                time.time() - step_st))
    #             step_st = time.time()
    #
    #     accuracy = accuracy_score(all_y_true, all_y_pred)
    #     precision = precision_score(all_y_true, all_y_pred, average='macro')
    #     recall = recall_score(all_y_true, all_y_pred, average='macro')
    #     F1_score = f1_score(all_y_true, all_y_pred, average='macro')
    #
    #     print('epoch: {}/{}, Epoch testing time {:.2f}s, Total testing time {:.2f}s'.format(epoch, args.total_epochs,
    #                                                                                         time.time() - epoch_st,
    #                                                                                         time.time() - test_st))
    #     f2_test_metrics.append([accuracy, precision, recall, F1_score])
    #     record = "model-{:03d}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}\n".format(epoch,
    #                                                                                                         accuracy,
    #                                                                                                         precision,
    #                                                                                                         recall,
    #                                                                                                         F1_score)
    #     print(record)
    #     with open("./f2_classification_metric_record.txt", 'a') as fp:
    #         fp.write(record)

    print('Start Testing...{}'.format(all_test_file))
    iteration = all_test_samples // batch_size
    test_st = time.time()
    total_iter = 0
    all_test_metrics = []
    for epoch in range(1, args.total_epochs + 1):
        epoch_st = time.time()
        print('Loading model......')
        st = time.time()
        print('Restore from model.ckpt-{}'.format(epoch))
        saver.restore(sess, './checkpoint/model/model.ckpt-{}'.format(epoch))
        print('Finished loading, {:.2f}s\n'.format(time.time() - st))

        step_st = time.time()
        all_y_pred = []
        all_y_true = []
        for step in range(1, iteration + 1):
            start_time = time.time()
            batch_x, batch_y = sess.run([all_test_x, all_test_y])
            test_feed_dict = {
                x: batch_x,
                label: batch_y,
                training_flag: False
            }
            pred = sess.run([logits_list], feed_dict=test_feed_dict)
            np_pred = np.asarray(pred)
            # softmax_x_all = np.asarray(pred).reshape(-1).tolist()
            # softmax_x_all = softmax(softmax_x_all)
            # softmax_x_all = softmax_x_all.reshape(-1, 1)
            # mean prediction
            mean_pred = np.mean(np_pred[0], axis= 0)

            # softmax_x = np.asarray(mean_pred).reshape(-1).tolist()
            # softmax_x = softmax(softmax_x)
            # softmax_x = softmax_x.reshape(-1, 1)

            y_pred = np.argmax(mean_pred[0], 0)
            y_true = np.argmax(batch_y[0], 0)
            all_y_pred.append(y_pred)
            all_y_true.append(y_true)
            end_time = time.time()
            # print("consuming time:" + str(end_time-start_time))
            total_iter += 1
            if step % 100 == 0:
                print('Ep {}, Iter {}, Step {}/{}, {:.2f}s'.format(epoch, total_iter, step, iteration,
                                                                   time.time() - step_st))
                step_st = time.time()

        accuracy = accuracy_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred, average='macro')
        recall = recall_score(all_y_true, all_y_pred, average='macro')
        F1_score = f1_score(all_y_true, all_y_pred, average='macro')

        print('epoch: {}/{}, Epoch testing time {:.2f}s, Total testing time {:.2f}s'.format(epoch, args.total_epochs,
                                                                                            time.time() - epoch_st,
                                                                                            time.time() - test_st))

        record = "model-{:03d}, accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}\n".format(epoch,
                                                                                                            accuracy,
                                                                                                            precision,
                                                                                                            recall,
                                                                                                            F1_score)
        print(record)
        with open("./all_classification_metric_record.txt", 'a') as fp:
            fp.write(record)


    coord.request_stop()
    coord.join(threads)
