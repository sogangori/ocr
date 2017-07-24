import tensorflow as tf
import tensorflow.contrib.slim as slim

modelName = "./weights/model_dense.pd"
initializer = tf.contrib.layers.xavier_initializer()
regularizer = slim.l2_regularizer(0.0005)

def denseBlock(input_layer,i,j,channel):
    with tf.variable_scope("dense_unit"+str(i)):
        nodes = []
        a = slim.conv2d(input_layer,channel,[3,3],normalizer_fn=slim.batch_norm, 
                        weights_initializer = initializer, weights_regularizer= regularizer)
        nodes.append(a)
        for z in range(j):
            src_concat = a
            for k in range(len(nodes)):
                src_concat = tf.concat([src_concat,nodes[k]],3)
            print ('src_concat',z,src_concat)
            b = slim.conv2d(src_concat,channel,[3,3],normalizer_fn=slim.batch_norm)
            nodes.append(b)
        return b

def inference(src, train):
    LABEL = 2350
    channel = 32
    units_between_stride = 3
    layer1 = slim.conv2d(src,channel,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))
    for i in range(2):
        layer1 = denseBlock(layer1,i,units_between_stride,channel)
        layer1 = slim.conv2d(layer1,channel,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
    
    top = slim.conv2d(layer1,channel,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')

    out = slim.layers.flatten(top)
    out = slim.layers.dropout(out, 0.5)
    out = slim.layers.fully_connected(out,LABEL)
    return out; 