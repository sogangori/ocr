import tensorflow as tf
import model.model_helper as helper

modelName = "./weights/model_vgg11_s.pd"
LABEL = 2350
channel = 64

def inference(src, train):
    
    pool = tf.layers.conv2d(src,filters=channel, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = helper.max_pool_2(pool)    

    pool = tf.layers.conv2d(pool,filters=channel*2, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = helper.max_pool_2(pool)    

    pool = tf.layers.conv2d(pool,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = helper.max_pool_2(pool)    
    
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = helper.max_pool_2(pool)    

    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = helper.max_pool_2(pool)   

    #pool = tf.nn.dropout(pool, 0.5)
    pool = helper.reshape_4d_to_2d(pool)
    pool = tf.layers.dense(pool,units=LABEL, activation=tf.nn.relu , use_bias=True)
            
    return pool; 