from __future__ import print_function

import numpy as np
import tensorflow  as tf

from tensorflow.examples.tutorials.mnist import input_data

#Import MNIST data
mnist=input_data.read_data_sets('F:tensortflow\mnist.data',one_hot=True)
#Uses a little mnist data
Xtr,Ytr=mnist.train.next_batch(5000)
Xte,Yte=mnist.train.next_batch(200)

#tf Graph input
xtr=tf.placeholder("float",[None,784])
xte=tf.placeholder('float',[784])
#negative function :y=-x  arg_mix(array,axis)
#nearest neighbor calculation using L1 Distance
distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),reduction_indices=1)
#get the min  distance nn_index
pred_index=tf.argmin(distance,0)
accuracy=0
with tf.Session() as sess:
    #initializer all variables
    init=tf.global_variables_initializer()
    sess.run(init)
    #start training
    for i in range(len(Xte)):
        xtr_index=sess.run(pred_index,feed_dict={xtr:Xtr,xte:Xte[i,:]})

        print("test",i,"prediction:",np.argmax(Ytr[xtr_index]),"true class :",np.argmax(Yte[i]))
        if np.argmax(Ytr[nn_index])==np.argmax(Yte[i]):
            accuracy+=1.0/len(Xte)
    print("Done")

    print("Accuracy:",accuracy)
