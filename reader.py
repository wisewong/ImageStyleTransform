from os import listdir
from os.path import isfile, join

import tensorflow as tf


def get_image(path, height, width, preprocess_fn):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)


def batch_image(batch_size, height, width, path, preprocess_fn, epochs=2, shuffle=True):
    file_names = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        file_names = sorted(file_names)

    png = file_names[0].lower().endswith('png')

    filename_queue = tf.train.string_input_producer(file_names, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess_fn(image, height, width)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)
