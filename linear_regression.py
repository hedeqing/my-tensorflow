from __future__ import print_function

import tensorflow as tf
import numpy  as np
import matplotlib.pyplot as plt
rng=np.random

#Parameters

learning_rate =0.01
training_epochs=1000
display_step=50

#Training Data
train_X=np.asarray([3.3,4.4,5.5,6.71,6.93])
train_Y=np.asarray([1.7,2.76,2.09,3.19,1.694])
n_samples=train_X.shape[0]

#tf Graph Input
X=tf.placeholder('float')
Y=tf.placeholder('float')

#Set weights
w=tf.Variable(rng.randn(),name='weight')
b=tf.Variable(rng.randn(),name='bias')

#Construct a linear model
pre=tf.add(tf.multiply(X,w),b)

#Mean squared error  (pre-Y)**2/daa.shape[0]
cost =tf.reduce_sum(tf.pow(pre-Y,2))/(2*n_samples)

#GradientDescentOptimizer descent
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Initialize the variables
init=tf.global_variables_initializer()

#Start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        #diaplay logs per epoch
        if (epoch+1)% display_step ==0:
            c= sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("epoch :",'%04d' % (epoch +1),"cost=","{:.9f}".format(c),'w=',sess.run(w),'b=',sess.run(b))

    print("optimizer Finished")
    training_cost=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
    print("training cost =",training_cost,'w=',sess.run(w),'b=',sess.run(b),'\n')

    #Graph display
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(w)*train_X+sess.run(b),label="Fitted line")
    plt.legend()
    plt.show()

    #GradientDescentOptimizer Examples as requested
    test_X=np.asarray([6.83,4.668,8.9,7.91])
    test_Y=np.asarray([1.84,2.273,3.3,2.841])
    print("Testing  (mean square loss Comparison)")
    testing_cost=sess.run(tf.reduce_sum(tf.pow(pre-Y,2))/(2*test_X.shape[0]),feed={X:test_X,Y:test_Y})
    print("Testing cost=",testing_cost)
    print('Absolute mean squared loss difference',abs(training_cost-testing_cost))
    plt.plot(test_X,test_Y,"ro",label='label original data')
    plt.plot(train_X,sess.run(w)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
