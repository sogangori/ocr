import tensorflow as tf
import model.model_helper as helper

#acc(0.985,0.928)
modelName = "./weights/model_vgg11.pd"
LABEL = 2350
channel = 64

with tf.variable_scope('vgg11'):
    w11c  = tf.get_variable("conv1_1", shape=[3, 3, 1, channel], initializer =tf.contrib.layers.xavier_initializer())
    w11b = tf.Variable(tf.constant(0.0, shape=[channel]))
    
    w21c = tf.get_variable("conv2_1", shape=[3, 3, channel, channel*2], initializer =tf.contrib.layers.xavier_initializer())
    w21b = tf.Variable(tf.constant(0.0, shape=[channel*2]))
    
    w31c = tf.get_variable("conv3_1", shape=[3, 3, channel*2, channel*4], initializer =tf.contrib.layers.xavier_initializer())
    w31b = tf.Variable(tf.constant(0.0, shape=[channel*4]))

    w32c = tf.get_variable("conv3_2", shape=[3, 3, channel*4, channel*4], initializer =tf.contrib.layers.xavier_initializer())
    w32b = tf.Variable(tf.constant(0.0, shape=[channel*4]))

    w41c = tf.get_variable("conv4_1", shape=[3, 3, channel*4, channel*8], initializer =tf.contrib.layers.xavier_initializer())
    w41b = tf.Variable(tf.constant(0.0, shape=[channel*8]))

    w42c = tf.get_variable("conv4_2", shape=[3, 3, channel*8, channel*8], initializer =tf.contrib.layers.xavier_initializer())
    w42b = tf.Variable(tf.constant(0.0, shape=[channel*8]))

    w51c = tf.get_variable("conv5_1", shape=[3, 3, channel*8, channel*8], initializer =tf.contrib.layers.xavier_initializer())
    w51b = tf.Variable(tf.constant(0.0, shape=[channel*8]))

    w52c = tf.get_variable("conv5_2", shape=[3, 3, channel*8, channel*8], initializer =tf.contrib.layers.xavier_initializer())
    w52b = tf.Variable(tf.constant(0.0, shape=[channel*8]))

    w1f = tf.Variable(tf.random_normal([2048,LABEL]))    
    w1b = tf.Variable(tf.constant(0.0, shape=[LABEL]))

    w2f = tf.Variable(tf.random_normal([4096,LABEL]))    
    w2b = tf.Variable(tf.constant(0.0, shape=[LABEL]))

def inference(src, train):
    
    #pool = helper.Gaussian_noise_Add(pool, 0.1, 0.3)

    pool = helper.conv2dRelu(src,w11c,w11b)
    pool = helper.max_pool_2(pool)    

    pool = helper.conv2dRelu(pool,w21c,w21b)    
    pool = helper.max_pool_2(pool)
        
    pool = helper.conv2dRelu(pool,w31c,w31b)
    pool = helper.conv2dRelu(pool,w32c,w32b)   
    pool = helper.max_pool_2(pool)    

    pool = helper.conv2dRelu(pool,w41c,w41b)
    pool = helper.conv2dRelu(pool,w42c,w42b)   
    pool = helper.max_pool_2(pool)

    pool = helper.conv2dRelu(pool,w51c,w51b)
    pool = helper.conv2dRelu(pool,w52c,w52b)   
    pool = helper.max_pool_2(pool)

    pool = helper.reshape_4d_to_2d(pool)
    pool = tf.nn.dropout(pool, 0.5)
    pool = tf.matmul(pool, w1f)    
    pool = tf.nn.bias_add(pool, w1b)

    #pool = tf.nn.dropout(pool, 0.9)
    #pool = tf.matmul(pool, w2f)    
    #pool = tf.nn.bias_add(pool, w2b)
        
    return pool; 