import tensorflow as tf
def adam(lr):
    return tf.train.AdamOptimizer( learning_rate=lr, epsilon=0.1, decay=0.9, name="adamOpt" )

def rms(lr):
    return tf.train.RMSPropOptimizer( learning_rate=lr, epsilon=1.0,  decay=0.9, name="RMSpropOpt" )

def sgd(lr):
    return tf.train.GradientDescentOptimizer( learning_rate=lr, name="SGDOpt" )

def mom(lr):
    return tf.train.MomentumOptimizer( learning_rate=lr, momentum=0.99, name="MomeOpt" )

def adagrad(lr):
    return tf.train.AdagradOptimizer( learning_rate=lr, initial_accumulator_value = 0.9, name='adagradOpt')
