import tensorflow as tf

def regularizer():
    regula=0    
    for var in tf.trainable_variables():        
        regula += tf.cond(tf.rank(var) > 2, lambda: tf.constant(0.0), lambda: tf.nn.l2_loss(var))
    return regula
