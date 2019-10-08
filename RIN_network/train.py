import tensorflow.contrib.slim as slim
import argparse
import os
import sys
import time

from utils.utils import ReDirectSTD
from utils.read_decode import *
from utils.network import PIL
from utils.data_aug import data_augmentation


######################################################################
# Options
# --------
parser = argparse.ArgumentParser()
parser.add_argument('--record_dir', type=str, default='./datasets/blister_pack_RTT/datas/tfrecords/')
parser.add_argument('--feature_dir', type=str, default='./datasets/blister_pack_RTT/features/')
parser.add_argument('--model_dir', type=str, default='./checkpoint/model/')
parser.add_argument('--pre_model', type=str, default='')
parser.add_argument('--restore_model', type=str, default='')
parser.add_argument('--dataset', type=str, default='blister_pack_RTT')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--reduction_ratio', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001)  # 0.001
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--drop_rate', type=float, default=0.5)
parser.add_argument('--part', type=bool, default=True)
parser.add_argument('--all', type=bool, default=True)
parser.add_argument('--half', type=bool, default=False)
parser.add_argument('--straight', type=bool, default=False)
parser.add_argument('--only_test', type=bool, default=False)
parser.add_argument('--load_model', type=bool, default=True)
parser.add_argument('--feat_ext', type=bool, default=True)
parser.add_argument('--normalize_feat', type=bool, default=True)
parser.add_argument('--rerank', type=bool, default=True)
args = parser.parse_args()

######################################################################
# Settings
# --------
train_file = args.record_dir + 'train.tfrecords'

class_num = 230
train_samples = 19314 #21*2*230

iteration = train_samples // args.batch_size


reduction_ratio = args.reduction_ratio
batch_size = args.batch_size
drop_rate = args.drop_rate
folder_now = args.model_dir + time.strftime("%y%m%d_%H%M%S", time.localtime())

ReDirectSTD(folder_now + 'stdout.txt', 'stdout', True)

######################################################################
# Main
# --------
###################
# Loading Dataset #
###################
# training data
print('train:', train_file)
train_x, train_y = read_and_decode_t(train_file, 16)
train_y = tf.one_hot(train_y, class_num)  # rearrange training label

###########
# Setting #
###########
augmentation = True
if augmentation:
    train_x = tf.cast(train_x, dtype=tf.float32)
    train_x = tf.image.random_saturation(train_x, lower=0.5, upper=1.5)
    train_x = tf.image.random_contrast(train_x, lower=0.5, upper=1.5)
    train_x = tf.image.random_brightness(train_x, max_delta=0.2)

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])  # input image size
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

local_x_list, logits_list = PIL(x, training=training_flag, class_num=class_num, drop_rate=args.drop_rate,
                                reduction_ratio=args.reduction_ratio,
                                part=args.part, straight=args.straight, half=args.half, all=args.all).model
loss_list = []
accuracy_list = []
for logits in logits_list:
    # print('shape of loss logits: ', np.shape(logits))
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits)
    loss_list.append(loss)
    # print('shape of loss : ', np.shape(loss_list))
    correct_prediction_p = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy_p = tf.reduce_mean(tf.cast(correct_prediction_p, tf.float32))
    accuracy_list.append(accuracy_p)

cost = tf.reduce_sum(loss_list)
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=args.momentum, use_nesterov=True)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train = optimizer.minimize(cost + l2_loss * args.weight_decay)

accuracy = tf.reduce_mean(tf.cast(accuracy_list, tf.float32))

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

variables = slim.get_variables_to_restore()
# for v in variables:
#     print(v.name.split('/')[0])

# ------chose which you want to restore------
# variables_to_restore2 = [v for v in variables if v.name.split('/')[0] != 'part_fully_connected_0'
#                          and v.name.split('/')[0] != 'part_fully_connected_1'
#                          and v.name.split('/')[0] != 'part_fully_connected_2'
#                          and v.name.split('/')[0] != 'part_fully_connected_3'
#                          and v.name.split('/')[0] != 'part_fully_connected_4'
#                          and v.name.split('/')[0] != 'part_fully_connected_5'
#                          and v.name.split('/')[0] != 'all_fully_connected'
#                          and v.name.split('/')[0] != 'half_fully_connected_0'
#                          and v.name.split('/')[0] != 'half_fully_connected_1'
#                          and v.name.split('/')[0] != 'straight_fully_connected_0'
#                          and v.name.split('/')[0] != 'straight_fully_connected_1'
#                          and v.name.split('/')[0] != 'straight_fully_connected_2']
# for v in variables_to_restore2:
#     print(v)
# variables_to_restore2 = [v for v in variables if v.name.split('_')[0] != 'all'
#                         and v.name.split('_')[0] != 'half'
#                         and v.name.split('_')[0] != 'straight']
# variables_to_restore2 = [v for v in variables if v.name.split('_')[0] != 'part']
# variables_to_restore3 = [v for v in variables if v.name.split('_')[0] == 'all']
# variables_to_restore4 = [v for v in variables if v.name.split('_')[0] == 'half']
# variables_to_restore5 = [v for v in variables if v.name.split('_')[0] == 'straight']
# saver2 = tf.train.Saver(variables_to_restore2)
# saver3 = tf.train.Saver(variables_to_restore3)
# saver4 = tf.train.Saver(variables_to_restore4)
# saver5 = tf.train.Saver(variables_to_restore5)


# with tf.Session() as sess:
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print('Initialing all variables......')
    # sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print('Finished initialing.')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if args.only_test:
        if args.load_model:
            print('Loading model......')
            st = time.time()
            print('Restore from ', args.restore_model)
            saver.restore(sess, args.restore_model)
            print('Finished loading, {:.2f}s\n'.format(time.time() - st))

        # test(val=False)

        coord.request_stop()
        coord.join(threads)
        sys.exit("Finish testing")

    if args.pre_model != '':
        print('Start loading......')
        st = time.time()
        print('Restore from ', args.pre_model)
        saver.restore(sess, args.pre_model)

        print('Finished loading, {:.2f}s\n'.format(time.time() - st))

    if not os.path.exists(folder_now):
        os.makedirs(folder_now)

    # -----plot the figures on tensorboard------
    train_writer = tf.summary.FileWriter(folder_now + '/logs/train', sess.graph)


    #############
    # Training #
    ############
    print('Start Training......')
    train_st = time.time()
    total_iter = 0
    epoch_learning_rate = args.learning_rate
    for epoch in range(1, args.total_epochs + 1):
        epoch_st = time.time()
        print('Epoch: ', epoch)

        if epoch % 60 == 0:
            epoch_learning_rate = epoch_learning_rate / 10

        train_acc = 0.0
        train_loss = 0.0

        step_st = time.time()
        for step in range(1, iteration + 1):
            batch_x, batch_y = sess.run([train_x, train_y])

            batch_x = data_augmentation(batch_x)  # flip images and translation

            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag: True
            }

            _, batch_loss, batch_acc = sess.run([train, cost, accuracy], feed_dict=train_feed_dict)

            total_iter += 1
            if step % 100 == 0:
                print('Ep {}, Iter {}, Step {}/{}, {:.2f}s, batch_loss: {:.4f}, batch_acc: {:.4f}'.format(
                    epoch, total_iter, step, iteration, time.time() - step_st, batch_loss, batch_acc))
                step_st = time.time()

            train_loss += batch_loss
            train_acc += batch_acc

        train_loss /= iteration  # average loss
        train_acc /= iteration  # average accuracy

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='accuracy', simple_value=train_acc)])

        print('Epoch training time {:.2f}s, Total training time {:.2f}s'.format(
            time.time() - epoch_st, time.time() - train_st))

        train_writer.add_summary(summary=train_summary, global_step=epoch)
        train_writer.flush()

        print('epoch: {}/{}, train_loss: {:.4f}, train_acc: {:.4f}'.format(
            epoch, args.total_epochs, train_loss, train_acc))

        print('\nSaving model to ', folder_now + '\n')
        saver.save(sess=sess, save_path=folder_now + '/model.ckpt', global_step=epoch)

    coord.request_stop()
    coord.join(threads)