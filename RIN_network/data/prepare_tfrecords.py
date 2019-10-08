# coding=utf-8
import os
import tensorflow as tf
from PIL import Image

dataset_path = '.././datasets/blister_pack_RTT/datas/'
tf_path = dataset_path + 'tfrecords/'


def create_record2(path, tfrecords_name):
    print('start')
    writer = tf.python_io.TFRecordWriter(tf_path + tfrecords_name)

    for img_name in os.listdir(path):
        print(img_name)
        img_name_split = img_name.split('_')
        real_id = int(img_name_split[0])
        cam_num = int(img_name_split[1])
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        img = img.resize((224, 224))  # width, height
        # print(np.shape(img))
        img_raw = img.tobytes()  # 将图片转化为原生bytes

        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[real_id])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'cam_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[cam_num])),
            'real_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[real_id]))
        }))
        writer.write(example.SerializeToString())
    print('Finish')
    writer.close()


if __name__ == '__main__':
    create_record2('.././datasets/blister_pack_RTT/images/train', 'train.tfrecords')
    create_record2('.././datasets/blister_pack_RTT/images/test/all/', 'test_all.tfrecords')