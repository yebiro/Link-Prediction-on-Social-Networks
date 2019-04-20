import tensorflow as tf
import numpy as np

def weight_variable_glorot(input_dim, output_dim, dtype=tf.float32, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    #返回input_dim*output_dim的矩阵，产生于minval和maxval之间，产生的值是均匀分布的。
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=dtype)
    return tf.Variable(initial, name=name)
