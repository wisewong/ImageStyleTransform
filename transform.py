# coding: utf-8
from __future__ import print_function

import os
import time

import tensorflow as tf

import model
import reader
from preprocessing import preprocessing_factory

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                                                   'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "models.ckpt", "风格模型")
tf.app.flags.DEFINE_string("image_file", "content.jpg", "输入图片")
tf.app.flags.DEFINE_string('target_file', 'res.jpg', '转换风格后的图片')

FLAGS = tf.app.flags.FLAGS


def main(_):
    height = 0
    width = 0
    with open(FLAGS.image_file, 'rb') as img:
        with tf.Session().as_default() as sess:
            if FLAGS.image_file.lower().endswith('png'):
                image = sess.run(tf.image.decode_png(img.read()))
            else:
                image = sess.run(tf.image.decode_jpeg(img.read()))
            height = image.shape[0]
            width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height))

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                FLAGS.loss_model,
                is_training=False)
            """获取经过预处理的输入图片，用于后面获取图片的content"""
            image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
            image = tf.expand_dims(image, 0)
            generated = model.transform_network(image, training=False)
            generated = tf.squeeze(generated, [0])
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            """获取已训练好的model"""
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)

            """生成转换style后的image"""
            start_time = time.time()
            generated = sess.run(generated)
            generated = tf.cast(generated, tf.uint8)
            end_time = time.time()
            tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

            generated_file = FLAGS.target_file
            if os.path.exists('static/img/generated') is False:
                os.makedirs('static/img/generated')
            with open(generated_file, 'wb') as img:
                img.write(sess.run(tf.image.encode_jpeg(generated)))
                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
