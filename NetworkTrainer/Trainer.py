
import time
import numpy as np
from time import localtime, strftime
import sklearn.metrics as metrics
from six.moves import urllib
from six.moves import xrange
import numpy as np
import tensorflow as tf
import model.model_layer7 as model 
from util.data_reader import DataReader
import util.trainer_helper as helper

DataReader = DataReader()
EVAL_FREQUENCY = 2
NUM_EPOCHS = 1000
isNewTrain = True      

def main(argv=None):        
      
  trainIn,trainOut,testIn,testOut = DataReader.GetData()
  trainIn = np.expand_dims(trainIn,axis=3)
  testIn = np.expand_dims(testIn,axis=3)
  trainIn = trainIn/255.0    
  testIn = testIn/255.0
  iter_count = 4
  batch = (int)(trainIn.shape[0]/iter_count)    
  X = tf.placeholder(tf.float32, [None,trainIn.shape[1],trainIn.shape[2],1])
  Y = tf.placeholder(tf.int32, [None])
  IsTrain = tf.placeholder(tf.bool)  
    
  predict = model.inference(X, IsTrain)
  argMax = tf.cast( tf.arg_max(predict,1), tf.int32)     
  print('argMax',argMax)
  acc = tf.reduce_mean(tf.cast(tf.equal(argMax, Y), tf.float32))
  entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = predict, labels = Y))  
  loss = entropy + 1e-5 * helper.regularizer()      
  LearningRate = 0.001    
  optimizer = tf.train.AdamOptimizer(LearningRate).minimize(loss) 
  
  with tf.Session() as sess:    
    tf.global_variables_initializer().run()    
    saver = tf.train.Saver()      
    if isNewTrain: print('Initialized!')
    else :        
        saver.restore(sess, model.modelName)
        print("Model restored")

    sess.run(tf.local_variables_initializer())  
        
    start_sec = start_time = time.time()
    for step in range(NUM_EPOCHS):
        
        for iter in range(iter_count):
            offset = iter * batch
            feed_dict = {X: trainIn[offset:offset+batch], Y: trainOut[offset:offset+batch], IsTrain:True}       
            _,l, accr = sess.run([optimizer,entropy, acc], feed_dict)          
            now = strftime("%H:%M:%S", localtime())
                                
        if step % EVAL_FREQUENCY == 0:            
            feed_dict_test = {X: testIn[offset:offset+batch], Y: testOut[offset:offset+batch], IsTrain :False}
            l_v, accr_v = sess.run([entropy,acc], feed_dict_test)        
            print('%d,%d, acc(%.3f,%.3f), entropy (%.4f,%.4f), %s' % (step,iter, accr,accr_v, l, l_v,now))             
                    
        this_sec = time.time()
        if this_sec - start_sec > 60 * 15 :
            start_sec = this_sec
            save_path = saver.save(sess, model.modelName)            
            print("Model Saved, time:%s" %(now))      
           
    print ('saver.save()', saver.save(sess, model.modelName))              
    

tf.app.run()