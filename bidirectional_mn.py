from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#import  mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("F:tensortflow\mnist.data",one_hot=True)
#Training Parameters
learning_rate=0.01
training_steps=1000
display_step=200
batch_size=200

#network Parameters
num_input=28  #MNIST data input(img shape=28*28)
timesteps=28
num_hidden=128
num_classes=10

#tf Graph input_data
X=tf.placeholder('float',[None,timesteps,num_input])
Y=tf.placeholder("float",[None,num_classes])

#Define weights
weights={'out':tf.Variable(tf.random_normal([2*num_hidden,num_classes]))}
biases={"out":tf.Variable(tf.random_normal([num_classes]))}

def BIRNN(x,weights,biases):
    #Prepare data shape to match 'rnn' function requirements
    #current data input shape:(batch_size,timesteps,num_input)
    #requirements shape: 'timesteps' tensor list of shape(batch_size,num_input)
    #unstack to get a list of 'timesteps' tensors of shape(batch_size,num_input)
    x=tf.unstack(x,timesteps,1)

    #Define lstm cellss with TensorFlow
    #forward direction cell
    lstm_fw_cell=rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)
    #backward direction cell
    lstm_bw_cell=rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)

    #Get lstm cell outputs
    try:
        outputs,_,_,=rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,
                                                    dtype=tf.float32)

    except Exception:
        outputs=rnn.static_bideirectional_rnn(lstm_fw_cell,lstm_bw_cell,x,
                                            dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']
logits=BIRNN(X,weights,biases)
prediction=tf.nn.softmax(logits)

#Define loss optimizer
loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)

#Evaluate model (wuith test logits ,for dropout to be disabled)
correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#initializer  the variables
init=tf.global_variables_initializer()

#Start training
with tf.Session() as sess:
    #run the initializer
    sess.run(init)

    for step in range(1,training_steps+1):
        batch_x,batch_y=mnist.train.next_batch(batch_size)
        #reshape data to get 28 seq of 28 elements
        batch_x=batch_x.reshape((batch_size,timesteps,num_input))
        #run optimizer op (backward)
        sess.run(train_op,feed_dict={X: batch_x, Y : batch_y})
        if step % display_step ==0 or step==1:
            loss,acc=sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y})
            print("step " +str(step)+",Minibatch loss =" + \
            "{:.4f}".format(loss) +",Training accrucy= "+"{:.3f}".format(acc))
    print("optimizer Finished")

    #Calculate accrucy for 128 mnist test images
    test_len=128
    test_data=mnist.test.images[:test_len].reshape((-1,timesteps,num_input))
    test_label=mnist.test.label[:test_len]
    print("Testing accrucy:",\
    sess.run(accuracy,feed_dict={X:test_data,Y:test_label}))
