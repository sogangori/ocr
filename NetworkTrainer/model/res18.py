import tensorflow as tf
import model.model_helper as helper

modelName = "./weights/model_res18.pd"
LABEL = 2350
channel = 64

def inference(src, train):
    #1
    pool = tf.layers.conv2d(src,filters=channel, kernel_size=[3, 3], strides=[2,2], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_1 = tf.layers.max_pooling2d(pool,pool_size=[3,3],strides=[2,2],padding= 'same')
    out_conv_1 = helper.LayerNormalize(out_conv_1)
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
    pool = tf.layers.conv2d(out_conv_2_2,filters=channel*4, kernel_size=[3, 3], strides=[2,2], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_2_2 = tf.layers.conv2d(out_conv_2_2,filters=channel*4, kernel_size=[1, 1], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_2_2 = tf.layers.max_pooling2d(out_conv_2_2,pool_size=[2,2],strides=[2,2],padding= 'same')
    out_conv_3_1 = tf.nn.relu(tf.add(out_conv_2_2, pool))
    #3_2
    pool = tf.layers.conv2d(out_conv_3_1,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*4, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_3_2 = tf.nn.relu(tf.add(out_conv_3_1, pool))
    out_conv_3_2= helper.LayerNormalize(out_conv_3_2)
    #4_1
    pool = tf.layers.conv2d(out_conv_3_2,filters=channel*8, kernel_size=[3, 3], strides=[2,2], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_3_2 = tf.layers.conv2d(out_conv_3_2,filters=channel*8, kernel_size=[1, 1], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_3_2 = tf.layers.max_pooling2d(out_conv_3_2,pool_size=[2,2],strides=[2,2],padding= 'same')
    out_conv_4_1 = tf.nn.relu(tf.add(out_conv_3_2, pool))
    #4_2
    pool = tf.layers.conv2d(out_conv_4_1,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_4_2 = tf.nn.relu(tf.add(out_conv_3_2, pool))
    #5_1
    pool = tf.layers.conv2d(out_conv_4_2,filters=channel*8, kernel_size=[3, 3], strides=[2,2], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_4_2 = tf.layers.conv2d(out_conv_4_2,filters=channel*8, kernel_size=[1, 1], strides=[2,2], padding="same",use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_4_2 = tf.layers.max_pooling2d(out_conv_4_2,pool_size=[2,2],strides=[2,2],padding= 'same')
    out_conv_5_1 = tf.nn.relu(tf.add(out_conv_4_2, pool))# 3,2 
    #5_2
    pool = tf.layers.conv2d(out_conv_5_1,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool = tf.layers.conv2d(pool,filters=channel*8, kernel_size=[3, 3], padding="same",activation=tf.nn.relu, use_bias=True, kernel_initializer=tf.contrib.layers.xavier_initializer())
    out_conv_5_2 = tf.nn.relu(tf.add(out_conv_5_1, pool))
    print ('out_conv_5_2',out_conv_5_2)

    pool = tf.layers.average_pooling2d(pool,pool_size=[2,2],strides=[2,2],padding= 'same')
    print ('avg out', pool._shape)
    #pool = tf.nn.dropout(pool, 0.5)
    pool = helper.reshape_4d_to_2d(pool)
    pool = tf.layers.dense(pool,units=LABEL, activation=tf.nn.relu , use_bias=True)
            
    return pool; 