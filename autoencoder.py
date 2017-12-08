from __future__ import division ,print_function,absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Import MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('F:tensortflow\mnist.data',one_hot=True)

#Training Parameters
learning_rate=0.01
num_steps=30000
batch_size=256

display_step=1000
examples_to_show=10

#Network Parameters

num_hidden_1=256
num_hidden_2=128
num_input=784

#tf Graph input
X=tf.placeholder('float',shape=[None,num_input])
weights={
    "encoder_h1":tf.Variable(tf.random_normal([num_input,num_hidden_1])),
    "encoder_h2":tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
    "decoder_h1":tf.Variable(tf.random_normal([num_hidden_2,num_hidden_1])),
    "decoder_h2":tf.Variable(tf.random_normal([num_hidden_1,num_input])),
    }
biases={
    "encoder_b1":tf.Variable(tf.random_normal([num_hidden_1])),
    "encoder_b2":tf.Variable(tf.random_normal([num_hidden_2])),
    "decoder_b1":tf.Variable(tf.random_normal([num_hidden_1])),
    "decoder_b2":tf.Variable(tf.random_normal([num_input])),
}
#Build the ancoder
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),
                                            biases["encoder_b1"]))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights["encoder_h2"]),
                                            biases["encoder_b2"]))
    return layer_2
#Build the decoder
def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),
                                            biases["decoder_b1"]))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights["decoder_h2"]),
                                            biases["decoder_b2"]))
    return layer_2
#Construct model

encoder_op=encoder(X)
decoder_op=decoder(encoder_op)
# Prediction
y_pred=decoder_op
#Targets (labels) are the input data
y_true=X
#Define loss and optimizer,minimize  the squared error (优化器：对梯度下降的优化)
loss =tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

#initializer the variables
init=tf.global_variables_initializer()

#start Training
with tf.Session() as sess:
    #run the initializer
    sess.run(init)

    #trianing
    for i in range(1,num_steps+1):
        batch_x,_=mnist.train.next_batch(batch_size)
        _,l=sess.run([optimizer,loss],feed_dict={X:batch_x})

        if i % display_step ==0 or i==1:
            print('step %i:  Minibatch loss: %f'% (i,l))
    #Testing
    #Encoder and Decoder images from test set and visualize the reconstruction
    n=4

    canvas_recon=np.empty((28*n,28*n))
    for i in range(n):
        #mnist  data set
        batch_x,_=mnist.test.next_bach(n)
        #Encoder and decoder the digit images
        g=ssess.run(decoder_op,feed_dict={X:batch_x})
        #display original images
        for j in range(n):
            #drae the original digits
            canvas_orig[i*28:(i+1)*28,j*28:(j+1)*28]=\
            batch_x[j].reshape([28,28])
        #display the reconstruct digits
        for j in range(n):
            #draw the reconstruct digit
            canvas_recon[i * 28:(i+1) * 28,j*28:(j+1) * 28]=\
            g[j].reshape([28,28])
    print("original images")
    plt.figure(figsize=(n,n))
    plt.imshow(canvas_orig,origin="upper",cmap="gray")
    plt.show()
    print("Reconstructed images")
    plt.figure(figsize=(n,n))
    plt.imshow(canvas_recon,origin="upper",cmap='gray')
    plt.show()
