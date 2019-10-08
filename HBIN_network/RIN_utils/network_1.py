import tensorflow as tf
from tflearn.layers.conv import global_avg_pool, avg_pool_2d, max_pool_2d
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

import numpy as np

######################################################################
# Model Definetions
# --------
def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride,
                                   padding=padding, name=layer_name)
        if activation:
            network = tf.nn.relu(network)
        return network

def Fully_connected(x, units, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        return tf.layers.dense(inputs=x, use_bias=True, units=units, name=layer_name)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers):
    return tf.concat(layers, axis=3)

# the part identity learning network
class PIL():
    def __init__(self, x, training, class_num, drop_rate, reduction_ratio, part, straight, all, half):
        self.training = training
        self.class_num = class_num
        self.drop_rate = drop_rate
        self.model = self.Build_SEnet(x, reduction_ratio, part, straight, all, half)

    def Stem(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=32, kernel=[3, 3], stride=2, padding='VALID', layer_name=scope + '_conv1')
            x = conv_layer(x, filter=32, kernel=[3, 3], padding='VALID', layer_name=scope + '_conv2')
            block_1 = conv_layer(x, filter=64, kernel=[3, 3], layer_name=scope + '_conv3')

            split_max_x = tf.layers.max_pooling2d(inputs=block_1, pool_size=[3, 3], strides=2, padding='VALID')
            split_conv_x = conv_layer(block_1, filter=96, kernel=[3, 3], stride=2, padding='VALID',
                                      layer_name=scope + '_split_conv1')
            x = Concatenation([split_max_x, split_conv_x])

            split_conv_x1 = conv_layer(x, filter=64, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x1 = conv_layer(split_conv_x1, filter=96, kernel=[3, 3], padding='VALID',
                                       layer_name=scope + '_split_conv3')

            split_conv_x2 = conv_layer(x, filter=64, kernel=[1, 1], layer_name=scope + '_split_conv4')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[7, 1], layer_name=scope + '_split_conv5')
            split_conv_x2 = conv_layer(split_conv_x2, filter=64, kernel=[1, 7], layer_name=scope + '_split_conv6')
            split_conv_x2 = conv_layer(split_conv_x2, filter=96, kernel=[3, 3], padding='VALID',
                                       layer_name=scope + '_split_conv7')

            x = Concatenation([split_conv_x1, split_conv_x2])

            split_conv_x = conv_layer(x, filter=192, kernel=[3, 3], stride=2, padding='VALID',
                                      layer_name=scope + '_split_conv8')
            split_max_x = tf.layers.max_pooling2d(inputs=x, pool_size=[3, 3], strides=2, padding='VALID')

            x = Concatenation([split_conv_x, split_max_x])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf.nn.relu(x)

            return x

    def Inception_resnet_A(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=32, kernel=[3, 3], layer_name=scope + '_split_conv3')

            split_conv_x3 = conv_layer(x, filter=32, kernel=[1, 1], layer_name=scope + '_split_conv4')
            split_conv_x3 = conv_layer(split_conv_x3, filter=48, kernel=[3, 3], layer_name=scope + '_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=64, kernel=[3, 3], layer_name=scope + '_split_conv6')

            x = Concatenation([split_conv_x1, split_conv_x2, split_conv_x3])
            x = conv_layer(x, filter=384, kernel=[1, 1], layer_name=scope + '_final_conv1', activation=False)

            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf.nn.relu(x)

            return x

    def Inception_resnet_B(self, x, scope):
        with tf.name_scope(scope):
            init = x

            split_conv_x1 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=128, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=160, kernel=[1, 7], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=192, kernel=[7, 1], layer_name=scope + '_split_conv4')

            x = Concatenation([split_conv_x1, split_conv_x2])
            x = conv_layer(x, filter=1152, kernel=[1, 1], layer_name=scope + '_final_conv1', activation=False)
            # 1154
            x = x * 0.1
            x = init + x

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf.nn.relu(x)

            return x

    # def Inception_resnet_C(self, x, scope):
    #     with tf.name_scope(scope):
    #         init = x
    #
    #         split_conv_x1 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv1')
    #
    #         split_conv_x2 = conv_layer(x, filter=192, kernel=[1, 1], layer_name=scope + '_split_conv2')
    #         split_conv_x2 = conv_layer(split_conv_x2, filter=224, kernel=[1, 3], layer_name=scope + '_split_conv3')
    #         split_conv_x2 = conv_layer(split_conv_x2, filter=256, kernel=[3, 1], layer_name=scope + '_split_conv4')
    #
    #         x = Concatenation([split_conv_x1, split_conv_x2])
    #         x = conv_layer(x, filter=2144, kernel=[1, 1], layer_name=scope + '_final_conv2', activation=False)
    #         # 2048
    #         x = x * 0.1
    #         x = init + x
    #
    #         x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
    #         x = tf.nn.relu(x)
    #
    #         return x

    def Reduction_A(self, x, scope):
        with tf.name_scope(scope):
            k = 256
            l = 256
            m = 384
            n = 384

            split_max_x = tf.layers.max_pooling2d(inputs=x, pool_size=[3, 3], strides=2, padding='VALID')

            split_conv_x1 = conv_layer(x, filter=n, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv1')

            split_conv_x2 = conv_layer(x, filter=k, kernel=[1, 1], layer_name=scope + '_split_conv2')
            split_conv_x2 = conv_layer(split_conv_x2, filter=l, kernel=[3, 3], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=m, kernel=[3, 3], stride=2, padding='VALID',
                                       layer_name=scope + '_split_conv4')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf.nn.relu(x)

            return x

    def Reduction_B(self, x, scope):
        with tf.name_scope(scope):
            split_max_x = tf.layers.max_pooling2d(inputs=x, pool_size=[3, 3], strides=1, padding='VALID')

            split_conv_x1 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv1')
            split_conv_x1 = conv_layer(split_conv_x1, filter=384, kernel=[3, 3], stride=1, padding='VALID',
                                       layer_name=scope + '_split_conv2')

            split_conv_x2 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv3')
            split_conv_x2 = conv_layer(split_conv_x2, filter=288, kernel=[3, 3], stride=1, padding='VALID',
                                       layer_name=scope + '_split_conv4')

            split_conv_x3 = conv_layer(x, filter=256, kernel=[1, 1], layer_name=scope + '_split_conv5')
            split_conv_x3 = conv_layer(split_conv_x3, filter=288, kernel=[3, 3], layer_name=scope + '_split_conv6')
            split_conv_x3 = conv_layer(split_conv_x3, filter=320, kernel=[3, 3], stride=1, padding='VALID',
                                       layer_name=scope + '_split_conv7')

            x = Concatenation([split_max_x, split_conv_x1, split_conv_x2, split_conv_x3])

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = tf.nn.relu(x)

            return x

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            squeeze = global_avg_pool(input_x, name='Global_avg_pooling')

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
            excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = input_x * excitation

            return scale

    # horizontal 6 parts
    def Part_extract(self, input_x, layer_name):
        with tf.name_scope(layer_name):
            output_height = int(np.shape(input_x)[1])
            output_weight = int(np.shape(input_x)[2])
            stripe_w = output_weight
            stripe_h = int(output_height / 6)
            local_x_list = []
            logits_list = []

            for i in range(6):  # 6 parts
                local_x = avg_pool_2d(input_x[:, i * stripe_h: (i + 1) * stripe_h, :, :], (stripe_h, stripe_w))
                # print('shape of local_x: ', np.shape(local_x))
                local_x = tf.layers.dropout(inputs=local_x, rate=self.drop_rate, training=self.training,
                                            name=layer_name + '_drop1')
                local_x = conv_layer(local_x, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv' + str(i))
                # print('shape of local_x after conv: ', np.shape(local_x))
                local_x = Batch_Normalization(local_x, training=self.training, scope=layer_name + '_batch' + str(i))
                local_x = flatten(local_x)
                # print('shape of local_x1: ', np.shape(local_x))
                local_x_list.append(local_x)
                local_x = tf.layers.dropout(inputs=local_x, rate=self.drop_rate, training=self.training,
                                            name=layer_name + '_drop2')
                local_x = Fully_connected(local_x, self.class_num, layer_name=layer_name + '_fully_connected_' + str(i))
                # print('shape of local_x after FC: ', np.shape(local_x))
                logits_list.append(local_x)
                # print('shape of logits_list: ', np.shape(logits_list))

            return local_x_list, logits_list

    # 1 part
    def All_extract(self, input_x, layer_name):
        with tf.name_scope(layer_name):
            output_height = int(np.shape(input_x)[1])
            output_weight = int(np.shape(input_x)[2])
            local_x_list = []
            logits_list = []
            # all fig
            local_x1 = avg_pool_2d(input_x[:, :, :, :], (output_height, output_weight), name=layer_name + '_AvgPool2D')
            # local_x1 = global_avg_pool(input_x, name='Global_avg_pooling')
            local_x1 = tf.layers.dropout(inputs=local_x1, rate=self.drop_rate, training=self.training,
                                         name=layer_name + '_drop1')
            local_x1 = conv_layer(local_x1, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv')
            local_x1 = Batch_Normalization(local_x1, training=self.training, scope=layer_name + '_batch')
            local_x1 = flatten(local_x1)
            local_x_list.append(local_x1)
            local_x1 = tf.layers.dropout(inputs=local_x1, rate=self.drop_rate, training=self.training,
                                         name=layer_name + '_drop2')
            local_x1 = Fully_connected(local_x1, self.class_num, layer_name=layer_name + '_fully_connected')
            logits_list.append(local_x1)

            return local_x_list, logits_list

    # horizontal 2 parts
    def Half_extract(self, input_x, layer_name):
        with tf.name_scope(layer_name):
            output_height = int(np.shape(input_x)[1])
            output_weight = int(np.shape(input_x)[2])
            stripe_w2 = output_weight
            stripe_h2 = int(output_height / 2)
            local_x_list = []
            logits_list = []
            for i in range(2):  # up and down
                local_x2 = avg_pool_2d(input_x[:, i * stripe_h2: (i + 1) * stripe_h2, :, :], (stripe_h2, stripe_w2))
                local_x2 = tf.layers.dropout(inputs=local_x2, rate=self.drop_rate, training=self.training,
                                             name=layer_name + '_drop1')
                local_x2 = conv_layer(local_x2, filter=256, kernel=[1, 1],
                                      layer_name=layer_name + '_split_conv' + str(i))
                local_x2 = Batch_Normalization(local_x2, training=self.training, scope=layer_name + '_batch' + str(i))
                local_x2 = flatten(local_x2)
                local_x_list.append(local_x2)
                local_x2 = tf.layers.dropout(inputs=local_x2, rate=self.drop_rate, training=self.training,
                                             name=layer_name + '_drop2')
                local_x2 = Fully_connected(local_x2, self.class_num, layer_name=layer_name + '_fully_connected_' + str(i))
                logits_list.append(local_x2)

            return local_x_list, logits_list

    # straight 3 parts
    def Straight_extract(self, input_x, layer_name):
        with tf.name_scope(layer_name):
            output_height = int(np.shape(input_x)[1])
            output_weight = int(np.shape(input_x)[2])
            stripe_w3 = int(output_weight / 3)
            stripe_h3 = output_height
            local_x_list = []
            logits_list = []
            # ***************Left Part*****************
            local_x3_1 = avg_pool_2d(input_x[:, :, :stripe_w3, :], (stripe_h3, stripe_w3))
            local_x3_1 = tf.layers.dropout(inputs=local_x3_1, rate=self.drop_rate, training=self.training,
                                           name=layer_name + '_drop11')
            local_x3_1 = conv_layer(local_x3_1, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv0')
            local_x3_1 = Batch_Normalization(local_x3_1, training=self.training, scope=layer_name + '_batch0')
            local_x3_1 = flatten(local_x3_1)
            local_x_list.append(local_x3_1)
            local_x3_1 = tf.layers.dropout(inputs=local_x3_1, rate=self.drop_rate, training=self.training,
                                           name=layer_name + '_drop12')
            local_x3_1 = Fully_connected(local_x3_1, self.class_num, layer_name=layer_name + '_fully_connected_0')
            logits_list.append(local_x3_1)

            # ***************Middle Part*****************
            if output_weight % 3 != 0:
                stripe_wm = int(stripe_w3 * 2 + output_weight % 3)
            else:
                stripe_wm = stripe_w3 * 2
            local_x3_2 = avg_pool_2d(input_x[:, :, stripe_w3:stripe_wm, :], (stripe_h3, stripe_wm))
            local_x3_2 = tf.layers.dropout(inputs=local_x3_2, rate=self.drop_rate, training=self.training,
                                           name=layer_name + '_drop21')
            local_x3_2 = conv_layer(local_x3_2, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv1')
            local_x3_2 = Batch_Normalization(local_x3_2, training=self.training, scope=layer_name + '_batch1')
            local_x3_2 = flatten(local_x3_2)
            local_x_list.append(local_x3_2)
            local_x3_2 = tf.layers.dropout(inputs=local_x3_2, rate=self.drop_rate, training=self.training,
                                           name=layer_name + '_drop22')
            local_x3_2 = Fully_connected(local_x3_2, self.class_num, layer_name=layer_name + '_fully_connected_1')
            logits_list.append(local_x3_2)

            # ***************Right Part*****************
            local_x3_3 = avg_pool_2d(input_x[:, :, stripe_wm:, :], (stripe_h3, stripe_w3))
            local_x3_3 = tf.layers.dropout(inputs=local_x3_3, rate=self.drop_rate, training=self.training,
                                           name=layer_name + '_drop31')
            local_x3_3 = conv_layer(local_x3_3, filter=256, kernel=[1, 1], layer_name=layer_name + '_split_conv2')
            local_x3_3 = Batch_Normalization(local_x3_3, training=self.training, scope=layer_name + '_batch2')
            local_x3_3 = flatten(local_x3_3)
            local_x_list.append(local_x3_3)
            local_x3_3 = tf.layers.dropout(inputs=local_x3_3, rate=self.drop_rate, training=self.training,
                                           name=layer_name + '_drop32')
            local_x3_3 = Fully_connected(local_x3_3, self.class_num, layer_name=layer_name + '_fully_connected_2')
            logits_list.append(local_x3_3)

            return local_x_list, logits_list

    # Building Network of Inception, Residual, Squeeze and Excitation,and parts
    def Build_SEnet(self, input_x, reduction_ratio, part, straight, all, half):
        input_x = tf.pad(input_x, [[0, 0], [32, 32], [32, 32], [0, 0]])
        # print('Input images size: ',np.shape(input_x))

        x = self.Stem(input_x, scope='stem')
        # print('Size after Stem: ', np.shape(x))

        for i in range(5):
            x = self.Inception_resnet_A(x, scope='Inception_A' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A' + str(i))

        x = self.Reduction_A(x, scope='Reduction_A')

        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_A')
        # print('Size after A: ', np.shape(x))

        for i in range(10):
            x = self.Inception_resnet_B(x, scope='Inception_B' + str(i))
            channel = int(np.shape(x)[-1])
            x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B' + str(i))

        x = self.Reduction_B(x, scope='Reduction_B')

        channel = int(np.shape(x)[-1])
        x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_B')
        # print('Size after B: ', np.shape(x))

        # for i in range(5):
        #     x = self.Inception_resnet_C(x, scope='Inception_C' + str(i))
        #     channel = int(np.shape(x)[-1])
        #     x = self.Squeeze_excitation_layer(x, out_dim=channel, ratio=reduction_ratio, layer_name='SE_C' + str(i))
        # print('Size after C: ', np.shape(x))

        features_list = [] # features list for the feature extraction when testing
        pred_list = [] # the classification results after fully connected layer

        if part == True:
            local_x_list_6, logits_6 = self.Part_extract(x, layer_name='part')
            features_list.extend(local_x_list_6)
            pred_list.extend(logits_6)
            # print('shape of local_x_list_6: ', np.shape(local_x_list_6))
            # print('shape of logits_6: ', np.shape(logits_6))
            # print('shape of features_list-6: ', np.shape(features_list))
            # print('shape of pred_list-6: ', np.shape(pred_list))
        if all == True:
            local_x_list_1, logits_1 = self.All_extract(x, layer_name='all')
            features_list.extend(local_x_list_1)
            pred_list.extend(logits_1)
            # print('shape of local_x_list_1: ', np.shape(local_x_list_1))
            # print('shape of logits_1: ', np.shape(logits_1))
            # print('shape of features_list-1: ', np.shape(features_list))
            # print('shape of pred_list-1: ', np.shape(pred_list))
        if half == True:
            local_x_list_2, logits_2 = self.Half_extract(x, layer_name='half')
            features_list.extend(local_x_list_2)
            pred_list.extend(logits_2)
            # print('shape of local_x_list_2: ', np.shape(local_x_list_2))
            # print('shape of logits_2: ', np.shape(logits_2))
            # print('shape of features_list-2: ', np.shape(features_list))
            # print('shape of pred_list-2: ', np.shape(pred_list))
        if straight == True:
            local_x_list_3, logits_3 = self.Straight_extract(x, layer_name='straight')
            features_list.extend(local_x_list_3)
            pred_list.extend(logits_3)
            # print('shape of local_x_list_3: ', np.shape(local_x_list_3))
            # print('shape of logits_3: ', np.shape(logits_3))
            # print('shape of features_list-3: ', np.shape(features_list))
            # print('shape of pred_list-3: ', np.shape(pred_list))

        # logits = tf.reduce_mean(pred_list, 0)
        # print('shape of output logits: ', np.shape(logits))
        # print('shape of output pred_list: ', np.shape(pred_list))
        # print('shape of output features_list: ', np.shape(features_list))
        return features_list, pred_list