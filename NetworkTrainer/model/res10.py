import tensorflow as tf
import tensorflow.contrib.slim as slim

modelName = "./weights/model_res10.pd"

def resUnit(input_layer,i, channel):
    with tf.variable_scope("res_unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = tf.nn.relu(part1)
        part3 = slim.conv2d(part2,channel,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = tf.nn.relu(part4)
        part6 = slim.conv2d(part5,channel,[3,3],activation_fn=None)

        input_2x = slim.conv2d(input_layer ,channel,[3,3],activation_fn=None)
        output = input_2x + part6
        return output

def inference(src, train):
    LABEL = 2350
    channel = 64
    
    layer1 = slim.conv2d(src,channel,[3,3],normalizer_fn=slim.batch_norm,scope='conv_'+str(0))

    for i in range(4):
        for j in range(1):
            layer1 = resUnit(layer1,j + (i*1),channel)
        layer1 = slim.conv2d(layer1,channel,[3,3],stride=[2,2],normalizer_fn=slim.batch_norm,scope='conv_s_'+str(i))
        print (i, layer1)
        channel*=2
    
    out = slim.conv2d(layer1,channel/2,[3,3],normalizer_fn=slim.batch_norm,activation_fn=None,scope='conv_top')
    out = slim.layers.flatten(out)    
    out = slim.layers.dropout(out, 0.5)
    out = slim.layers.fully_connected(out,LABEL)
    #output = slim.layers.softmax()
            
    return out; 