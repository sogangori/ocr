import tensorflow as tf
import model.model_helper as helper

modelName = "./weights/model_res14.pd"
LABEL = 2350
channel = 64

def inference(src, train):
    #1
    map = tf.layers.conv2d(src,filters=channel, kernel_size=[3, 3], strides=[2,2], padding="same", use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    map = tf.layers.batch_normalization(map)
    out_conv_1 = tf.nn.relu(map)
    #out_conv_1 = tf.layers.max_pooling2d(map,pool_size=[3,3],strides=[2,2],padding= 'same')    
    #2_1
    pool = tf.layers.conv2d(out_conv_1,filters=channel*1, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*1, kernel_size=[3, 3], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())    
    out_conv_2_1 = tf.nn.relu(tf.add(out_conv_1, pool))
    #2_2
    pool = tf.layers.conv2d(out_conv_2_1,filters=channel*2, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*2, kernel_size=[3, 3], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_2_1 = tf.layers.conv2d(out_conv_2_1,filters=channel*2, kernel_size=[1, 1], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_2_2 = tf.nn.relu(tf.add(out_conv_2_1, pool))
    #3_1
    pool = tf.layers.conv2d(out_conv_2_2,filters=channel*4, kernel_size=[3, 3], strides=[2,2], padding="same", use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.batch_normalization(pool)
    pool = tf.nn.relu(pool)
    pool = tf.layers.conv2d(pool,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_2_2 = tf.layers.conv2d(out_conv_2_2,filters=channel*4, kernel_size=[1, 1], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_2_2 = tf.layers.max_pooling2d(out_conv_2_2,pool_size=[2,2],strides=[2,2],padding= 'same')
    out_conv_3_1 = tf.nn.relu(tf.add(out_conv_2_2, pool))
    #3_2
    pool = tf.layers.conv2d(out_conv_3_1,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_3_2 = tf.nn.relu(tf.add(out_conv_3_1, pool))
          
    print ('avg in', out_conv_3_2._shape)
    pool = tf.layers.average_pooling2d(out_conv_3_2,pool_size=[12,12],strides=[12,12],padding= 'same')
    print ('avg out', pool._shape)
    pool = tf.nn.dropout(pool, 0.9)
    pool = helper.reshape_4d_to_2d(pool)
    pool = tf.layers.dense(pool,units=LABEL, activation=tf.nn.relu , use_bias=True)
            
    return pool; 