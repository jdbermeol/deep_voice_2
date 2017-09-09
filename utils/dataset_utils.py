import tensorflow as tf
import numpy as np
import pickle as pkl


def build_dataset(sess, inputs, names, batch_size, shuffle_buffer_size):
    placeholders = tuple([tf.placeholder(inp.dtype, inp.shape) for inp in inputs])

    with tf.device('/cpu:0'):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(placeholders)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()

        batch_inputs = iterator.get_next()
        batch_inputs = {na: inp for na, inp in zip(names, batch_inputs)}
        sess.run(iterator.initializer, feed_dict=dict(zip(placeholders, inputs)))
    return batch_inputs
