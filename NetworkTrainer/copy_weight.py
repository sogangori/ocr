import numpy as np
import tensorflow as tf
import model.vgg13 as model_src 
import model.vgg16 as model_dst 

#isCopy = True
isCopy = True

def main(argv=None):        

    variable_model_src = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'vgg13')
    variable_model_dst = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'vgg16')
    sess = tf.Session()
    saver_src = tf.train.Saver(variable_model_src)
    saver_dst = tf.train.Saver(variable_model_dst)          
    saver_src.restore(sess, model_src.modelName)
    print("model_src restored", model_src.modelName)
    print ('len src:dst = ',len(variable_model_src),len(variable_model_dst))

    if isCopy:
        #model_dst.w11c = tf.identity(variable_model_src[0])
        #model_dst.w11b = tf.identity(variable_model_src[1])      
          
        #model_dst.w11c = tf.Variable(tf.identity(variable_model_src[0]))
        #model_dst.w11b = tf.Variable(tf.identity(variable_model_src[1]))
        
        model_dst.w11c = tf.Variable(model_src.w11c.initialized_value(), name='conv1_1')    
        model_dst.w11b = tf.Variable(model_src.w11b.initialized_value())    

        #sess.run(tf.assign(model_dst.w11c, model_src.w11c))
        #sess.run(tf.assign(model_dst.w11b, model_src.w11b))

        sess.run(tf.global_variables_initializer())
        save_path = saver_dst.save(sess, model_dst.modelName)
        print ('save model_dst', save_path ) 
    else: 
        saver_dst.restore(sess, model_dst.modelName)
        print("model_dst restored", model_dst.modelName)

    print ('src',sess.run(tf.reduce_mean(model_src.w11c)),model_src.w11c)
    print ('src',sess.run(tf.reduce_mean(model_src.w11b)),model_src.w11b)
    print ('dst',sess.run(tf.reduce_mean(model_dst.w11c)),model_dst.w11c)
    print ('dst',sess.run(tf.reduce_mean(model_dst.w11b)),model_dst.w11b)

    #we = tf.Variable(tf.constant(0.0, shape=[5]))
    



tf.app.run()