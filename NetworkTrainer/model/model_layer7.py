
import os
import sys
import tensorflow as tf
import model.model_helper as helper

modelName = "./weights/model_layer6.pd"
LABEL = 1700
channel = 36

with tf.variable_scope('smap'):
    w11c  = tf.get_variable("conv1_1", shape=[3, 3, 1, channel], initializer =tf.contrib.layers.xavier_initializer())
    w11b = tf.Variable(tf.constant(0.0, shape=[channel]))

    w12c = tf.get_variable("conv1_2", shape=[3, 3, channel, channel], initializer =tf.contrib.layers.xavier_initializer())
    w12b = tf.Variable(tf.constant(0.0, shape=[channel]))

    w21c = tf.get_variable("conv2_1", shape=[3, 3, channel, channel], initializer =tf.contrib.layers.xavier_initializer())
    w21b = tf.Variable(tf.constant(0.0, shape=[channel]))

    w22c = tf.get_variable("conv2_2", shape=[3, 3, channel, channel], initializer =tf.contrib.layers.xavier_initializer())
    w22b = tf.Variable(tf.constant(0.0, shape=[channel]))

    w31c = tf.get_variable("conv3_1", shape=[3, 3, channel, channel], initializer =tf.contrib.layers.xavier_initializer())
    w31b = tf.Variable(tf.constant(0.0, shape=[channel]))

    w32c = tf.get_variable("conv3_2", shape=[3, 3, channel, LABEL], initializer =tf.contrib.layers.xavier_initializer())
    w32b = tf.Variable(tf.constant(0.0, shape=[LABEL]))

    w41f = tf.Variable(tf.random_normal([6800,LABEL]))
    w41b = tf.Variable(tf.constant(0.0, shape=[LABEL]))

def inference(input, train):
    
    #pool = helper.Gaussian_noise_Add(pool, 0.1, 0.3)

    pool = helper.conv2dRelu(input,w11c,w11b)
    pool = helper.conv2dRelu(pool,w12c,w12b)   
    pool = helper.max_pool_2(pool)    

    pool = helper.conv2dRelu(pool,w21c,w21b)
    pool = helper.conv2dRelu(pool,w22c,w22b)   
    pool = helper.max_pool_2(pool)     
        
    pool = helper.conv2dRelu(pool,w31c,w31b)
    pool = helper.conv2dRelu(pool,w32c,w32b)   
    pool = helper.max_pool_2(pool)    

    shape = pool.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    pool = tf.reshape(pool, [-1, dim])
    pool = tf.matmul(pool, w41f)    
    pool = tf.nn.bias_add(pool, w41b)
        
    return pool; 